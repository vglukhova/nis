class MNISTDataLoader {

    async loadCSVFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onerror = () => reject(new Error('Cannot read file'));
            reader.onload = (e) => {
                try {
                    const lines = e.target.result
                        .replace(/\r\n/g, '\n').replace(/\r/g, '\n')
                        .split('\n').filter(l => l.trim() !== '');

                    const labels = [];
                    const pixels = [];

                    for (const line of lines) {
                        const vals = line.trim().split(',');
                        if (vals.length !== 785) continue;
                        const nums = vals.map(Number);
                        if (nums.some(isNaN)) continue;
                        labels.push(nums[0]);
                        pixels.push(nums.slice(1));
                    }

                    if (labels.length === 0) {
                        reject(new Error('No valid rows in CSV')); return;
                    }

                    // Normalise pixels → [0,1], shape [N,28,28,1]
                    // clone() forces a real memory copy so dispose of intermediates is safe
                    const raw = tf.tensor2d(pixels, [labels.length, 784]);
                    const xs  = raw.div(255).reshape([labels.length, 28, 28, 1]).clone();
                    raw.dispose();

                    // One-hot labels
                    const lraw = tf.tensor1d(labels, 'int32');
                    const ys   = tf.oneHot(lraw, 10).cast('float32').clone();
                    lraw.dispose();

                    resolve({ xs, ys, count: labels.length });
                } catch (err) { reject(err); }
            };
            reader.readAsText(file);
        });
    }

    async loadTrainFromFiles(file) {
        if (this.trainData) { this.trainData.xs.dispose(); this.trainData.ys.dispose(); }
        this.trainData = await this.loadCSVFile(file);
        return this.trainData;
    }

    async loadTestFromFiles(file) {
        if (this.testData) { this.testData.xs.dispose(); this.testData.ys.dispose(); }
        this.testData = await this.loadCSVFile(file);
        return this.testData;
    }

    // Returns INDEPENDENT tensors (not slices/views) by going through JS arrays.
    // This guarantees fit() cannot corrupt them.
    splitTrainVal(xs, ys, valRatio) {
        valRatio = valRatio || 0.1;
        const n    = xs.shape[0];
        const nVal = Math.floor(n * valRatio);
        const nTrn = n - nVal;
        // clone() ensures each slice is an independent tensor with its own memory
        return {
            trainXs: xs.slice([0,    0,0,0], [nTrn, 28,28,1]).clone(),
            trainYs: ys.slice([0,    0],     [nTrn, 10]).clone(),
            valXs:   xs.slice([nTrn, 0,0,0], [nVal, 28,28,1]).clone(),
            valYs:   ys.slice([nTrn, 0],     [nVal, 10]).clone(),
        };
    }

    // Add Gaussian noise — returns independent tensor, caller disposes
    addNoise(xs, stddev) {
        stddev = stddev || 0.3;
        const noise  = tf.randomNormal(xs.shape, 0, stddev);
        const noisy  = xs.add(noise).clipByValue(0, 1);
        noise.dispose();
        return noisy;
    }

    // Split noisy/clean into train/val using the same JS-array approach
    splitNoisyClean(clean, noisy, valRatio) {
        valRatio = valRatio || 0.1;
        const n    = clean.shape[0];
        const nVal = Math.floor(n * valRatio);
        const nTrn = n - nVal;
        // clone() gives each tensor independent GPU memory
        return {
            trnClean: clean.slice([0,    0,0,0], [nTrn, 28,28,1]).clone(),
            trnNoisy: noisy.slice([0,    0,0,0], [nTrn, 28,28,1]).clone(),
            valClean: clean.slice([nTrn, 0,0,0], [nVal, 28,28,1]).clone(),
            valNoisy: noisy.slice([nTrn, 0,0,0], [nVal, 28,28,1]).clone(),
        };
    }

    // Pick k random samples — returns independent tensors
    getRandomTestBatch(xs, ys, k) {
        k = k || 5;
        const n   = xs.shape[0];
        const idx = Array.from({length: n}, (_, i) => i);
        for (let i = n - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [idx[i], idx[j]] = [idx[j], idx[i]];
        }
        const sel = idx.slice(0, k);
        // tf.gather + clone() = safe independent tensors, no arraySync OOM
        return {
            batchXs: tf.gather(xs, sel).clone(),
            batchYs: tf.gather(ys, sel).clone(),
            indices: sel,
        };
    }

    // Draw single image [28][28][1] JS array to canvas
    drawToCanvas(imgArray, canvas, scale) {
        scale = scale || 4;
        const imgData = new ImageData(28, 28);
        for (let r = 0; r < 28; r++) {
            for (let c = 0; c < 28; c++) {
                const raw = imgArray[r][c];
                const val = Array.isArray(raw) ? raw[0] : raw;
                const v   = Math.min(255, Math.max(0, Math.round(val * 255)));
                const idx = (r * 28 + c) * 4;
                imgData.data[idx]     = v;
                imgData.data[idx + 1] = v;
                imgData.data[idx + 2] = v;
                imgData.data[idx + 3] = 255;
            }
        }
        const tmp = document.createElement('canvas');
        tmp.width = 28; tmp.height = 28;
        tmp.getContext('2d').putImageData(imgData, 0, 0);
        canvas.width  = 28 * scale;
        canvas.height = 28 * scale;
        const ctx = canvas.getContext('2d');
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(tmp, 0, 0, canvas.width, canvas.height);
    }

    dispose() {
        if (this.trainData) { this.trainData.xs.dispose(); this.trainData.ys.dispose(); this.trainData = null; }
        if (this.testData)  { this.testData.xs.dispose();  this.testData.ys.dispose();  this.testData  = null; }
    }
}
