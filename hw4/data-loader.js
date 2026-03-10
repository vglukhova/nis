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
                    // dataSync() forces GPU execution BEFORE dispose of raw
                    const raw = tf.tensor2d(pixels, [labels.length, 784]);
                    const divided = raw.div(255);
                    raw.dispose();
                    const xs = divided.reshape([labels.length, 28, 28, 1]);
                    // dataSync then re-tensor to guarantee xs owns its memory
                    const xsData = xs.dataSync();
                    divided.dispose();
                    const xsFinal = tf.tensor(xsData, [labels.length, 28, 28, 1], 'float32');

                    // One-hot labels
                    const lraw = tf.tensor1d(labels, 'int32');
                    const ysRaw = tf.oneHot(lraw, 10).cast('float32');
                    lraw.dispose();
                    const ysData = ysRaw.dataSync();
                    ysRaw.dispose();
                    const ys = tf.tensor(ysData, [labels.length, 10], 'float32');

                    resolve({ xs: xsFinal, ys: ys, count: labels.length });
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

        // Pull everything into JS arrays once
        const xsData = xs.arraySync();   // [N][28][28][1]
        const ysData = ys.arraySync();   // [N][10]

        const trnX = xsData.slice(0, nTrn);
        const trnY = ysData.slice(0, nTrn);
        const valX = xsData.slice(nTrn);
        const valY = ysData.slice(nTrn);

        return {
            trainXs: tf.tensor4d(trnX),
            trainYs: tf.tensor2d(trnY),
            valXs:   tf.tensor4d(valX),
            valYs:   tf.tensor2d(valY),
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

        const cData = clean.arraySync();
        const nData = noisy.arraySync();

        return {
            trnClean: tf.tensor4d(cData.slice(0, nTrn)),
            trnNoisy: tf.tensor4d(nData.slice(0, nTrn)),
            valClean: tf.tensor4d(cData.slice(nTrn)),
            valNoisy: tf.tensor4d(nData.slice(nTrn)),
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
        const sel    = idx.slice(0, k);
        const xsData = xs.arraySync();
        const ysData = ys.arraySync();
        return {
            batchXs: tf.tensor4d(sel.map(i => xsData[i])),
            batchYs: tf.tensor2d(sel.map(i => ysData[i])),
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
