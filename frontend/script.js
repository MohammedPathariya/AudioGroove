document.addEventListener('DOMContentLoaded', () => {
    const apiEndpoint = document.querySelector('meta[name="audiogroove-api"]')?.content;
    const healthEndpoint = apiEndpoint ? new URL('/', apiEndpoint).toString() : null;
    const healthTimeoutMs = 15_000;
    const healthRetryDelayMs = 5_000;
    const maxHealthAttempts = 4;
    const sampleSeeds = {
        'angry-chair': {
            name: 'Angry Chair',
            path: 'samples/angry-chair.mid',
            preview: 'samples/previews/angry-chair.wav',
        },
        'boom-boom-boom': {
            name: 'Boom Boom Boom',
            path: 'samples/boom-boom-boom.mid',
            preview: 'samples/previews/boom-boom-boom.wav',
        },
        'dam-that-river': {
            name: 'Dam That River',
            path: 'samples/dam-that-river.mid',
            preview: 'samples/previews/dam-that-river.wav',
        },
        'it-takes-me-away': {
            name: 'It Takes Me Away',
            path: 'samples/it-takes-me-away.mid',
            preview: 'samples/previews/it-takes-me-away.wav',
        },
        delicado: {
            name: 'Delicado',
            path: 'samples/delicado.mid',
            preview: 'samples/previews/delicado.wav',
        },
        'another-day': {
            name: 'Another Day',
            path: 'samples/another-day.mid',
            preview: 'samples/previews/another-day.wav',
        },
    };

    const navButtons = [...document.querySelectorAll('[data-view-target]')];
    const views = [...document.querySelectorAll('[data-view]')];
    const seedInputs = [...document.querySelectorAll('input[name="seed"]')];
    const previewButtons = [...document.querySelectorAll('[data-preview]')];
    const midiInput = document.getElementById('seed-midi-input');
    const selectionText = document.getElementById('selection-text');
    const generateButton = document.getElementById('generate-btn');
    const regenerateButton = document.getElementById('regenerate-btn');
    const progressContainer = document.getElementById('progress-container');
    const statusText = document.getElementById('status-text');
    const outputPanel = document.getElementById('output-panel');
    const downloadButton = document.getElementById('download-btn');
    const generationTimeNote = document.getElementById('generation-time-note');
    const backendStatus = document.getElementById('backend-status');
    const backendStatusText = document.getElementById('backend-status-text');

    let uploadedSeed = null;
    let lastSeed = null;
    let generatedMidiUrl = null;
    let previewAudio = null;
    let isGenerating = false;
    let backendState = 'waking';
    let healthCheckId = 0;
    let healthController = null;
    let healthRetryTimer = null;

    const setBackendState = (state) => {
        const labels = {
            online: 'Backend online',
            offline: 'Backend offline',
            waking: 'Backend waking up',
        };
        backendState = state;
        backendStatus.dataset.state = state;
        backendStatusText.textContent = labels[state];
    };

    const cancelBackendCheck = () => {
        healthCheckId += 1;
        if (healthController) {
            healthController.abort();
            healthController = null;
        }
        if (healthRetryTimer) {
            window.clearTimeout(healthRetryTimer);
            healthRetryTimer = null;
        }
    };

    const startBackendCheck = () => {
        cancelBackendCheck();
        const currentCheckId = healthCheckId;
        setBackendState('waking');

        const runAttempt = async (attempt) => {
            if (!healthEndpoint || currentCheckId !== healthCheckId) {
                if (!healthEndpoint) setBackendState('offline');
                return;
            }

            healthController = new AbortController();
            const timeout = window.setTimeout(() => healthController.abort(), healthTimeoutMs);

            try {
                const response = await fetch(healthEndpoint, {
                    cache: 'no-store',
                    signal: healthController.signal,
                });
                if (currentCheckId !== healthCheckId) return;

                if (!response.ok) {
                    setBackendState('offline');
                    return;
                }

                const health = await response.json();
                const isReady = health.status === 'ok' && health.model_loaded !== false;
                setBackendState(isReady ? 'online' : 'offline');
            } catch {
                if (currentCheckId !== healthCheckId) return;
                if (attempt + 1 < maxHealthAttempts) {
                    setBackendState('waking');
                    healthRetryTimer = window.setTimeout(() => runAttempt(attempt + 1), healthRetryDelayMs);
                } else {
                    setBackendState('offline');
                }
            } finally {
                window.clearTimeout(timeout);
                healthController = null;
            }
        };

        runAttempt(0);
    };

    const showView = (name) => {
        navButtons.forEach((button) => {
            const isActive = button.dataset.viewTarget === name;
            button.classList.toggle('is-active', isActive);
            button.setAttribute('aria-pressed', String(isActive));
        });
        views.forEach((view) => {
            const isActive = view.dataset.view === name;
            view.hidden = !isActive;
            view.classList.toggle('is-visible', isActive);
        });
        window.history.replaceState(null, '', `#${name}`);
    };

    const selectedSample = () => {
        const selectedInput = seedInputs.find((input) => input.checked);
        return selectedInput ? sampleSeeds[selectedInput.value] : null;
    };

    const resetOutput = () => {
        outputPanel.hidden = true;
        statusText.textContent = '';
        statusText.classList.remove('is-error');
    };

    const stopPreview = () => {
        if (previewAudio) {
            previewAudio.pause();
            previewAudio.currentTime = 0;
            previewAudio = null;
        }
        previewButtons.forEach((button) => {
            button.textContent = '▶ play';
            button.setAttribute('aria-pressed', 'false');
        });
    };

    const playPreview = async (sampleKey, button) => {
        const wasPlaying = button.getAttribute('aria-pressed') === 'true';
        stopPreview();
        if (wasPlaying) return;

        const sample = sampleSeeds[sampleKey];
        previewAudio = new Audio(sample.preview);
        previewAudio.addEventListener('ended', stopPreview, { once: true });
        previewAudio.addEventListener('error', () => {
            stopPreview();
            statusText.textContent = 'This preview could not be played.';
            statusText.classList.add('is-error');
        }, { once: true });

        try {
            await previewAudio.play();
            button.textContent = '■ stop';
            button.setAttribute('aria-pressed', 'true');
        } catch {
            stopPreview();
            statusText.textContent = 'Audio preview was blocked by the browser.';
            statusText.classList.add('is-error');
        }
    };

    const getSeedFile = async () => {
        if (uploadedSeed) return uploadedSeed;

        const sample = selectedSample();
        if (!sample) throw new Error('Choose a sample or upload a MIDI file.');

        const response = await fetch(sample.path);
        if (!response.ok) throw new Error('The selected sample could not be loaded.');
        const blob = await response.blob();
        return new File([blob], `${sample.name.toLowerCase().replaceAll(' ', '-')}.mid`, {
            type: 'audio/midi',
        });
    };

    const setGeneratingState = (generating) => {
        isGenerating = generating;
        generateButton.disabled = generating;
        regenerateButton.disabled = generating;
        generateButton.textContent = generating ? 'Generating melody…' : 'Continue this melody';
        generateButton.setAttribute('aria-busy', String(generating));
        progressContainer.hidden = !generating;
        generationTimeNote.hidden = !generating;
        if (generating) {
            outputPanel.hidden = true;
            statusText.textContent = 'Generating your melody…';
            statusText.classList.remove('is-error');
        }
    };

    const generateMusic = async (seedOverride = null) => {
        if (isGenerating) return;
        cancelBackendCheck();
        if (backendState !== 'online') setBackendState('waking');
        setGeneratingState(true);

        try {
            if (!apiEndpoint) throw new Error('The generation service is not configured.');
            const seedFile = seedOverride || await getSeedFile();
            lastSeed = seedFile;
            const formData = new FormData();
            formData.append('seed_midi', seedFile);

            const response = await fetch(apiEndpoint, { method: 'POST', body: formData });
            if (!response.ok) {
                let message = 'The generation service is unavailable.';
                try {
                    const errorData = await response.json();
                    message = errorData.error || message;
                } catch {
                    // The backend may return an HTML error page.
                }
                throw new Error(message);
            }

            const generatedMidi = await response.blob();
            setBackendState('online');
            if (generatedMidiUrl) URL.revokeObjectURL(generatedMidiUrl);
            generatedMidiUrl = URL.createObjectURL(generatedMidi);
            downloadButton.href = generatedMidiUrl;
            statusText.textContent = '';
            outputPanel.hidden = false;
        } catch (error) {
            statusText.textContent = error instanceof Error ? error.message : 'Could not generate the MIDI continuation.';
            statusText.classList.add('is-error');
            startBackendCheck();
        } finally {
            setGeneratingState(false);
        }
    };

    navButtons.forEach((button) => button.addEventListener('click', () => showView(button.dataset.viewTarget)));

    seedInputs.forEach((input) => input.addEventListener('change', () => {
        uploadedSeed = null;
        midiInput.value = '';
        selectionText.textContent = `Using “${sampleSeeds[input.value].name}”`;
        resetOutput();
    }));

    midiInput.addEventListener('change', () => {
        uploadedSeed = midiInput.files[0] || null;
        if (!uploadedSeed) return;
        seedInputs.forEach((input) => { input.checked = false; });
        selectionText.textContent = `Using “${uploadedSeed.name}”`;
        resetOutput();
    });

    previewButtons.forEach((button) => {
        button.setAttribute('aria-pressed', 'false');
        button.addEventListener('click', () => playPreview(button.dataset.preview, button));
    });

    generateButton.addEventListener('click', () => generateMusic());
    regenerateButton.addEventListener('click', () => generateMusic(lastSeed));

    const initialView = window.location.hash === '#project' ? 'project' : 'make';
    showView(initialView);
    startBackendCheck();

    window.addEventListener('beforeunload', () => {
        cancelBackendCheck();
        if (generatedMidiUrl) URL.revokeObjectURL(generatedMidiUrl);
        stopPreview();
    });
});
