document.addEventListener('DOMContentLoaded', () => {
    // --- Configuration ---
    const API_ENDPOINT = 'https://audiogroove-backend.onrender.com/generate';

    // --- DOM Elements ---
    const midiInputElement = document.getElementById('seed-midi-input');
    const fileNameElement = document.getElementById('file-name');
    const generateBtn = document.getElementById('generate-btn');
    const regenerateBtn = document.getElementById('regenerate-btn');
    const downloadBtn = document.getElementById('download-btn');
    const progressContainer = document.getElementById('progress-container');
    const statusText = document.getElementById('status-text');
    const musicIcon = document.querySelector('.music-icon');

    // --- State Management ---
    let isGenerating = false;
    let generatedMidiBlob = null;
    let lastUsedSeedFile = null;

    // --- Event Listeners ---
    midiInputElement.addEventListener('change', () => {
        if (midiInputElement.files.length > 0) {
            fileNameElement.textContent = midiInputElement.files[0].name;
        } else {
            fileNameElement.textContent = 'No file selected';
        }
    });

    generateBtn.addEventListener('click', () => {
        lastUsedSeedFile = midiInputElement.files[0]; // Can be undefined, which is fine
        handleGeneration(lastUsedSeedFile);
    });

    regenerateBtn.addEventListener('click', () => {
        // Regenerate with the same seed file as last time, or no file if none was used
        handleGeneration(lastUsedSeedFile);
    });

    downloadBtn.addEventListener('click', () => {
        if (!generatedMidiBlob) {
            alert('No music file to download!');
            return;
        }
        const url = URL.createObjectURL(generatedMidiBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'audiogroove_composition.mid';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    });

    // --- Core Generation Logic ---
    const handleGeneration = async (seedFile) => {
        if (isGenerating) return;
        isGenerating = true;
        setGeneratingUIState();

        const formData = new FormData();
        if (seedFile) {
            formData.append('seed_midi', seedFile);
        }

        try {
            const response = await fetch(API_ENDPOINT, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'An unknown error occurred.');
            }

            generatedMidiBlob = await response.blob();
            setSuccessUIState();

        } catch (error) {
            console.error('Generation Error:', error);
            setErrorUIState(error.message);
        } finally {
            isGenerating = false;
        }
    };

    // --- UI State Functions ---
    const setGeneratingUIState = () => {
        statusText.textContent = 'Crafting your musical masterpiece...';
        progressContainer.classList.remove('hidden');
        generateBtn.classList.add('hidden');
        regenerateBtn.classList.add('hidden');
        downloadBtn.classList.add('hidden');
        musicIcon.textContent = '⏳';
    };

    const setSuccessUIState = () => {
        statusText.textContent = 'Your composition is ready! 🎶';
        progressContainer.classList.add('hidden');
        generateBtn.classList.add('hidden');
        regenerateBtn.classList.remove('hidden');
        downloadBtn.classList.remove('hidden');
        musicIcon.textContent = '✅';
    };

    const setErrorUIState = (errorMessage) => {
        statusText.textContent = `Error: ${errorMessage}`;
        progressContainer.classList.add('hidden');
        generateBtn.classList.remove('hidden');
        regenerateBtn.classList.add('hidden');
        downloadBtn.classList.add('hidden');
        musicIcon.textContent = '❌';
    };
});