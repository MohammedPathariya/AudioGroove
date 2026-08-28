const parseMidiNotes = (arrayBuffer) => {
    const view = new DataView(arrayBuffer);
    const MIDI_HEADER = 0x4d546864;
    const TRACK_HEADER = 0x4d54726b;

    if (view.byteLength < 14 || view.getUint32(0) !== MIDI_HEADER) {
        throw new Error('The generated file is not valid MIDI.');
    }

    const headerLength = view.getUint32(4);
    const trackCount = view.getUint16(10);
    const ticksPerBeat = view.getUint16(12);
    if (ticksPerBeat & 0x8000) {
        throw new Error('This MIDI timing format is not supported in the browser player.');
    }

    const noteEvents = [];
    const tempoEvents = [{ tick: 0, microsecondsPerBeat: 500_000, order: -1 }];
    let offset = 8 + headerLength;
    let eventOrder = 0;
    let finalTick = 0;

    const readVariableLength = (state, limit) => {
        let value = 0;
        let byte = 0;
        do {
            if (state.offset >= limit) throw new Error('The MIDI file ended unexpectedly.');
            byte = view.getUint8(state.offset);
            state.offset += 1;
            value = (value << 7) | (byte & 0x7f);
        } while (byte & 0x80);
        return value;
    };

    for (let trackIndex = 0; trackIndex < trackCount; trackIndex += 1) {
        if (offset + 8 > view.byteLength || view.getUint32(offset) !== TRACK_HEADER) {
            throw new Error('The generated MIDI track is invalid.');
        }

        const trackLength = view.getUint32(offset + 4);
        const trackEnd = offset + 8 + trackLength;
        if (trackEnd > view.byteLength) throw new Error('The generated MIDI track is incomplete.');

        const state = { offset: offset + 8 };
        let tick = 0;
        let runningStatus = null;

        while (state.offset < trackEnd) {
            tick += readVariableLength(state, trackEnd);
            finalTick = Math.max(finalTick, tick);

            let status = view.getUint8(state.offset);
            if (status & 0x80) {
                state.offset += 1;
                if (status < 0xf0) runningStatus = status;
            } else if (runningStatus !== null) {
                status = runningStatus;
            } else {
                throw new Error('The generated MIDI uses invalid running status.');
            }

            if (status === 0xff) {
                if (state.offset >= trackEnd) throw new Error('The generated MIDI metadata is incomplete.');
                const metaType = view.getUint8(state.offset);
                state.offset += 1;
                const metaLength = readVariableLength(state, trackEnd);
                if (state.offset + metaLength > trackEnd) throw new Error('The generated MIDI metadata is invalid.');
                if (metaType === 0x51 && metaLength === 3) {
                    tempoEvents.push({
                        tick,
                        microsecondsPerBeat: (view.getUint8(state.offset) << 16)
                            | (view.getUint8(state.offset + 1) << 8)
                            | view.getUint8(state.offset + 2),
                        order: eventOrder,
                    });
                }
                state.offset += metaLength;
                eventOrder += 1;
                continue;
            }

            if (status === 0xf0 || status === 0xf7) {
                const sysexLength = readVariableLength(state, trackEnd);
                state.offset += sysexLength;
                if (state.offset > trackEnd) throw new Error('The generated MIDI system event is invalid.');
                eventOrder += 1;
                continue;
            }

            const command = status & 0xf0;
            const channel = status & 0x0f;
            const dataLength = command === 0xc0 || command === 0xd0 ? 1 : 2;
            if (state.offset + dataLength > trackEnd) throw new Error('The generated MIDI event is incomplete.');
            const data1 = view.getUint8(state.offset);
            const data2 = dataLength === 2 ? view.getUint8(state.offset + 1) : 0;
            state.offset += dataLength;

            if (command === 0x90 && data2 > 0) {
                noteEvents.push({ type: 'on', tick, channel, note: data1, velocity: data2, order: eventOrder });
            } else if (command === 0x80 || (command === 0x90 && data2 === 0)) {
                noteEvents.push({ type: 'off', tick, channel, note: data1, velocity: 0, order: eventOrder });
            }
            eventOrder += 1;
        }

        offset = trackEnd;
    }

    tempoEvents.sort((a, b) => a.tick - b.tick || a.order - b.order);
    const tempoPoints = [];
    let activeTempo = 500_000;
    let previousTick = 0;
    let elapsedSeconds = 0;
    tempoEvents.forEach((event) => {
        elapsedSeconds += ((event.tick - previousTick) * activeTempo) / (ticksPerBeat * 1_000_000);
        previousTick = event.tick;
        activeTempo = event.microsecondsPerBeat;
        const point = { tick: event.tick, seconds: elapsedSeconds, microsecondsPerBeat: activeTempo };
        if (tempoPoints.at(-1)?.tick === event.tick) tempoPoints[tempoPoints.length - 1] = point;
        else tempoPoints.push(point);
    });

    const tickToSeconds = (tick) => {
        let point = tempoPoints[0];
        for (let index = 1; index < tempoPoints.length && tempoPoints[index].tick <= tick; index += 1) {
            point = tempoPoints[index];
        }
        return point.seconds + ((tick - point.tick) * point.microsecondsPerBeat) / (ticksPerBeat * 1_000_000);
    };

    noteEvents.sort((a, b) => a.tick - b.tick || a.order - b.order);
    const activeNotes = new Map();
    const notes = [];
    noteEvents.forEach((event) => {
        const key = `${event.channel}:${event.note}`;
        const pending = activeNotes.get(key) || [];
        if (event.type === 'on') {
            pending.push(event);
            activeNotes.set(key, pending);
            return;
        }
        const startEvent = pending.shift();
        if (!startEvent) return;
        if (pending.length === 0) activeNotes.delete(key);
        const start = tickToSeconds(startEvent.tick);
        const end = tickToSeconds(event.tick);
        notes.push({
            channel: event.channel,
            note: event.note,
            velocity: startEvent.velocity,
            start,
            duration: Math.max(0.06, end - start),
        });
    });

    activeNotes.forEach((pending) => pending.forEach((event) => {
        const start = tickToSeconds(event.tick);
        notes.push({
            channel: event.channel,
            note: event.note,
            velocity: event.velocity,
            start,
            duration: Math.max(0.12, tickToSeconds(finalTick) - start),
        });
    }));

    if (notes.length === 0) throw new Error('The generated MIDI does not contain playable notes.');
    return notes;
};

class MidiBrowserPlayer {
    constructor(onStateChange) {
        this.onStateChange = onStateChange;
        this.context = null;
        this.nodes = new Set();
        this.stopTimer = null;
        this.isPlaying = false;
    }

    async play(arrayBuffer) {
        this.stop();
        const notes = parseMidiNotes(arrayBuffer);
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (!AudioContextClass) throw new Error('This browser does not support audio playback.');

        this.context = new AudioContextClass();
        await this.context.resume();
        const startTime = this.context.currentTime + 0.08;
        const masterGain = this.context.createGain();
        masterGain.gain.value = 0.32;
        masterGain.connect(this.context.destination);

        let playbackDuration = 0;
        notes.forEach((midiNote) => {
            const oscillator = this.context.createOscillator();
            const noteGain = this.context.createGain();
            const noteStart = startTime + midiNote.start;
            const noteEnd = noteStart + Math.min(midiNote.duration, 12);
            const level = Math.max(0.012, (midiNote.velocity / 127) * (midiNote.channel === 9 ? 0.025 : 0.055));

            oscillator.type = midiNote.channel === 9 ? 'square' : 'triangle';
            oscillator.frequency.value = 440 * (2 ** ((midiNote.note - 69) / 12));
            noteGain.gain.setValueAtTime(0.0001, noteStart);
            noteGain.gain.exponentialRampToValueAtTime(level, noteStart + Math.min(0.025, midiNote.duration / 3));
            noteGain.gain.exponentialRampToValueAtTime(0.0001, noteEnd);
            oscillator.connect(noteGain);
            noteGain.connect(masterGain);
            oscillator.addEventListener('ended', () => this.nodes.delete(oscillator), { once: true });
            oscillator.start(noteStart);
            oscillator.stop(noteEnd + 0.02);
            this.nodes.add(oscillator);
            playbackDuration = Math.max(playbackDuration, midiNote.start + Math.min(midiNote.duration, 12));
        });

        this.isPlaying = true;
        this.onStateChange(true);
        this.stopTimer = window.setTimeout(() => this.stop(), (playbackDuration + 0.2) * 1000);
    }

    stop() {
        if (this.stopTimer) window.clearTimeout(this.stopTimer);
        this.stopTimer = null;
        this.nodes.forEach((node) => {
            try { node.stop(); } catch { /* The note may already have ended. */ }
        });
        this.nodes.clear();
        if (this.context) this.context.close();
        this.context = null;
        if (this.isPlaying) this.onStateChange(false);
        this.isPlaying = false;
    }
}

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
            preview: 'samples/previews/angry-chair.mp3',
        },
        'boom-boom-boom': {
            name: 'Boom Boom Boom',
            path: 'samples/boom-boom-boom.mid',
            preview: 'samples/previews/boom-boom-boom.mp3',
        },
        'dam-that-river': {
            name: 'Dam That River',
            path: 'samples/dam-that-river.mid',
            preview: 'samples/previews/dam-that-river.mp3',
        },
        'it-takes-me-away': {
            name: 'It Takes Me Away',
            path: 'samples/it-takes-me-away.mid',
            preview: 'samples/previews/it-takes-me-away.mp3',
        },
        delicado: {
            name: 'Delicado',
            path: 'samples/delicado.mid',
            preview: 'samples/previews/delicado.mp3',
        },
        'another-day': {
            name: 'Another Day',
            path: 'samples/another-day.mid',
            preview: 'samples/previews/another-day.mp3',
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
    const playGeneratedButton = document.getElementById('play-generated-btn');
    const generationTimeNote = document.getElementById('generation-time-note');
    const backendStatus = document.getElementById('backend-status');
    const backendStatusText = document.getElementById('backend-status-text');

    let uploadedSeed = null;
    let lastSeed = null;
    let generatedMidiUrl = null;
    let generatedMidiBuffer = null;
    let previewAudio = null;
    let isGenerating = false;
    let backendState = 'waking';
    let healthCheckId = 0;
    let healthController = null;
    let healthRetryTimer = null;
    const generatedPlayer = new MidiBrowserPlayer((playing) => {
        playGeneratedButton.textContent = playing ? '■ Stop playback' : '▶ Play in browser';
        playGeneratedButton.setAttribute('aria-pressed', String(playing));
    });

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
        generatedPlayer.stop();
        generatedMidiBuffer = null;
        if (generatedMidiUrl) URL.revokeObjectURL(generatedMidiUrl);
        generatedMidiUrl = null;
        downloadButton.removeAttribute('href');
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

        generatedPlayer.stop();
        const sample = sampleSeeds[sampleKey];
        previewAudio = new Audio(new URL(sample.preview, document.baseURI).href);
        previewAudio.preload = 'auto';
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
        if (generating) generatedPlayer.stop();
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
            generatedMidiBuffer = await generatedMidi.arrayBuffer();
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
    playGeneratedButton.addEventListener('click', async () => {
        if (generatedPlayer.isPlaying) {
            generatedPlayer.stop();
            return;
        }
        if (!generatedMidiBuffer) return;

        stopPreview();
        statusText.textContent = '';
        statusText.classList.remove('is-error');
        try {
            await generatedPlayer.play(generatedMidiBuffer);
        } catch (error) {
            generatedPlayer.stop();
            statusText.textContent = error instanceof Error ? error.message : 'The generated MIDI could not be played.';
            statusText.classList.add('is-error');
        }
    });

    const initialView = window.location.hash === '#project' ? 'project' : 'make';
    showView(initialView);
    startBackendCheck();

    window.addEventListener('beforeunload', () => {
        cancelBackendCheck();
        if (generatedMidiUrl) URL.revokeObjectURL(generatedMidiUrl);
        generatedPlayer.stop();
        stopPreview();
    });
});
