class MNISTApp {
    constructor() {
        this.loader      = new MNISTDataLoader();
        this.trainData   = null;
        this.testData    = null;
        this.classifier  = null;
        this.aeMax       = null;
        this.aeAvg       = null;
        this.isTraining  = false;
        this.noiseStddev = 0.3;
        this.initUI();
    }

    initUI() {
        document.getElementById('loadDataBtn').addEventListener('click',         () => this.onLoadData());
        document.getElementById('trainClassifierBtn').addEventListener('click',  () => this.onTrainClassifier());
        document.getElementById('evaluateBtn').addEventListener('click',         () => this.onEvaluate());
        document.getElementById('trainAutoencoderBtn').addEventListener('click', () => this.onTrainAutoencoder());
        document.getElementById('testFiveBtn').addEventListener('click',         () => this.onTestFive());
        document.getElementById('saveBtn').addEventListener('click',             () => this.onSave());
        document.getElementById('loadModelBtn').addEventListener('click',        () => this.onLoadModel());
        document.getElementById('resetBtn').addEventListener('click',            () => this.onReset());
        document.getElementById('toggleVisorBtn').addEventListener('click',      () => tfvis.visor().toggle());

        const slider = document.getElementById('noiseSlider');
        const lbl    = document.getElementById('noiseVal');
        slider.addEventListener('input', () => {
            this.noiseStddev = parseFloat(slider.value);
            lbl.textContent  = this.noiseStddev.toFixed(2);
        });
        lbl.textContent = this.noiseStddev.toFixed(2);
    }

    // ── Load Data ─────────────────────────────────────────────────────────────

    async onLoadData() {
        const trainFile = document.getElementById('trainFile').files[0];
        const testFile  = document.getElementById('testFile').files[0];
        if (!trainFile || !testFile)
            return this.log('ERROR: Select both Train and Test CSV files.');
        try {
            this.log('Loading train CSV…');
            this.trainData = await this.loader.loadTrainFromFiles(trainFile);
            this.log('Loading test CSV…');
            this.testData  = await this.loader.loadTestFromFiles(testFile);
            this.log('Data loaded — train: ' + this.trainData.count + ', test: ' + this.testData.count);
            document.getElementById('dataStatus').innerHTML =
                '<b>Train:</b> ' + this.trainData.count + ' samples<br>' +
                '<b>Test:</b> '  + this.testData.count  + ' samples';
        } catch (err) { this.log('ERROR: ' + err.message); }
    }

    // ── Classifier ────────────────────────────────────────────────────────────

    async onTrainClassifier() {
        if (!this.trainData) return this.log('ERROR: Load data first.');
        if (this.isTraining) return this.log('ERROR: Already training.');
        this.isTraining = true;
        try {
            if (this.classifier) { this.classifier.dispose(); this.classifier = null; }
            this.classifier = this.buildClassifier();
            this.log('Training classifier…');

            // splitTrainVal returns INDEPENDENT tensors (not slices)
            const sp = this.loader.splitTrainVal(this.trainData.xs, this.trainData.ys, 0.1);

            const t0 = Date.now();
            const hist = await this.classifier.fit(sp.trainXs, sp.trainYs, {
                epochs: 5, batchSize: 128,
                validationData: [sp.valXs, sp.valYs],
                shuffle: true,
                callbacks: tfvis.show.fitCallbacks(
                    { name: 'Classifier', tab: 'Training' },
                    ['loss', 'val_loss', 'acc', 'val_acc'],
                    { callbacks: ['onEpochEnd'] }
                )
            });

            sp.trainXs.dispose(); sp.trainYs.dispose();
            sp.valXs.dispose();   sp.valYs.dispose();

            const best = Math.max.apply(null, hist.history.val_acc);
            this.log('Classifier done in ' + ((Date.now()-t0)/1000).toFixed(1) +
                     's  |  best val_acc: ' + best.toFixed(4));
            this.updateModelInfo();
        } catch (err) {
            this.log('ERROR: ' + err.message);
        } finally { this.isTraining = false; }
    }

    async onEvaluate() {
        if (!this.classifier) return this.log('ERROR: Train or load a classifier first.');
        if (!this.testData)   return this.log('ERROR: Load test data first.');
        try {
            this.log('Evaluating…');
            const preds  = this.classifier.predict(this.testData.xs);
            const predLb = preds.argMax(-1);
            const trueLb = this.testData.ys.argMax(-1);
            const acc    = (await predLb.equal(trueLb).mean().data())[0];
            this.log('Test accuracy: ' + (acc * 100).toFixed(2) + '%');

            const predArr = await predLb.array();
            const trueArr = await trueLb.array();
            const cm = Array.from({length:10}, () => new Array(10).fill(0));
            for (let i = 0; i < predArr.length; i++) cm[trueArr[i]][predArr[i]]++;
            tfvis.render.confusionMatrix(
                { name: 'Confusion Matrix', tab: 'Evaluation' },
                { values: cm, tickLabels: ['0','1','2','3','4','5','6','7','8','9'] }
            );
            preds.dispose(); predLb.dispose(); trueLb.dispose();
        } catch (err) { this.log('ERROR: ' + err.message); }
    }

    // ── Autoencoders ──────────────────────────────────────────────────────────

    async onTrainAutoencoder() {
        if (!this.trainData) return this.log('ERROR: Load data first.');
        if (this.isTraining) return this.log('ERROR: Already training.');
        this.isTraining = true;
        try {
            if (this.aeMax) { this.aeMax.dispose(); this.aeMax = null; }
            if (this.aeAvg) { this.aeAvg.dispose(); this.aeAvg = null; }

            this.aeMax = this.buildAutoencoder('max');
            this.aeAvg = this.buildAutoencoder('avg');

            const N    = this.trainData.xs.shape[0];
            const nVal = Math.floor(N * 0.1);
            const nTrn = N - nVal;

            // Step 1: extract clean slices - keep alive the whole time
            const cleanTrn = this.trainData.xs.slice([0,    0,0,0], [nTrn, 28,28,1]);
            const cleanVal = this.trainData.xs.slice([nTrn, 0,0,0], [nVal, 28,28,1]);

            // Step 2: build noisy versions - keep alive the whole time
            const noise1 = tf.randomNormal(cleanTrn.shape, 0, this.noiseStddev);
            const noisyTrn = cleanTrn.add(noise1).clipByValue(0,1);
            noise1.dispose();

            const noise2 = tf.randomNormal(cleanVal.shape, 0, this.noiseStddev);
            const noisyVal = cleanVal.add(noise2).clipByValue(0,1);
            noise2.dispose();

            // Diagnostic: confirm noisy != clean
            const diff = noisyTrn.sub(cleanTrn);
            const meanDiff = diff.abs().mean();
            const diffVal = (await meanDiff.data())[0];
            diff.dispose(); meanDiff.dispose();
            this.log('Mean |noisy-clean| = ' + diffVal.toExponential(3) + ' (should be ~0.17 for stddev=0.3)');

            if (diffVal < 0.001) {
                this.log('ERROR: noisy and clean are identical — data problem!');
                cleanTrn.dispose(); cleanVal.dispose();
                noisyTrn.dispose(); noisyVal.dispose();
                return;
            }

            this.log('Train: ' + nTrn + ', Val: ' + nVal + ' samples');

            const self = this;
            const makeCb = (name) => {
                const cb = tfvis.show.fitCallbacks(
                    { name: name, tab: 'Autoencoders' },
                    ['loss', 'val_loss'],
                    { callbacks: ['onEpochEnd'] }
                );
                const orig = cb.onEpochEnd ? cb.onEpochEnd.bind(cb) : null;
                cb.onEpochEnd = async (epoch, logs) => {
                    self.log(name + ' ep'+(epoch+1)+': loss='+logs.loss.toExponential(3)+' val_loss='+logs.val_loss.toExponential(3));
                    if (orig) await orig(epoch, logs);
                };
                return cb;
            };

            this.log('Training Max-Pooling Autoencoder…');
            let t = Date.now();
            await this.aeMax.fit(noisyTrn, cleanTrn, {
                epochs: 15, batchSize: 64,
                validationData: [noisyVal, cleanVal],
                shuffle: true,
                callbacks: makeCb('AE Max Pooling')
            });
            this.log('Max-AE done in ' + ((Date.now()-t)/1000).toFixed(1) + 's');

            this.log('Training Avg-Pooling Autoencoder…');
            t = Date.now();
            await this.aeAvg.fit(noisyTrn, cleanTrn, {
                epochs: 15, batchSize: 64,
                validationData: [noisyVal, cleanVal],
                shuffle: true,
                callbacks: makeCb('AE Avg Pooling')
            });
            this.log('Avg-AE done in ' + ((Date.now()-t)/1000).toFixed(1) + 's');

            // Dispose ALL tensors only after training is fully done
            cleanTrn.dispose(); cleanVal.dispose();
            noisyTrn.dispose(); noisyVal.dispose();

            this.updateModelInfo();
            this.log('Both autoencoders trained. Click "Test 5 Random" to see results.');
        } catch (err) {
            this.log('ERROR: ' + err.message);
            console.error(err);
        } finally { this.isTraining = false; }
    }

    // ── Test 5 Random ─────────────────────────────────────────────────────────

    async onTestFive() {
        if (!this.testData)             return this.log('ERROR: Load test data first.');
        if (!this.aeMax || !this.aeAvg) return this.log('ERROR: Train autoencoders first.');
        try {
            const { batchXs, batchYs } = this.loader.getRandomTestBatch(
                this.testData.xs, this.testData.ys, 5
            );

            const noisy  = this.loader.addNoise(batchXs, this.noiseStddev);
            const outMax = this.aeMax.predict(noisy);
            const outAvg = this.aeAvg.predict(noisy);

            // Labels
            const lblT   = batchYs.argMax(-1);
            const lblArr = await lblT.array();
            lblT.dispose();

            // To JS arrays BEFORE any dispose
            const cleanArr = await batchXs.array();
            const noisyArr = await noisy.array();
            const maxArr   = await outMax.array();
            const avgArr   = await outAvg.array();

            batchXs.dispose(); batchYs.dispose();
            noisy.dispose(); outMax.dispose(); outAvg.dispose();

            this.renderPreview(cleanArr, noisyArr, maxArr, avgArr, lblArr);
        } catch (err) { this.log('ERROR: ' + err.message); }
    }

    renderPreview(cleanArr, noisyArr, maxArr, avgArr, lblArr) {
        const container = document.getElementById('previewContainer');
        container.innerHTML = '';

        const rows = [
            { label: 'Original Clean',           color: '#444',    data: cleanArr },
            { label: 'Noisy Input (σ = ' + this.noiseStddev.toFixed(2) + ')', color: '#444',    data: noisyArr },
            { label: 'Max Pooling – Denoised', color: '#1565C0', data: maxArr   },
            { label: 'Avg Pooling – Denoised', color: '#1565C0', data: avgArr   },
        ];

        rows.forEach((row, ri) => {
            if (ri === 2) {
                const hdr = document.createElement('div');
                hdr.className = 'cmp-header';
                hdr.textContent = 'Compare Max Pooling vs Average Pooling';
                container.appendChild(hdr);
            }
            const block = document.createElement('div');
            block.className = 'preview-block';

            const lbl = document.createElement('div');
            lbl.className = 'row-label';
            lbl.style.color = row.color;
            lbl.textContent = row.label;
            block.appendChild(lbl);

            const rowEl = document.createElement('div');
            rowEl.className = 'preview-row';

            row.data.forEach((imgData, i) => {
                const item   = document.createElement('div');
                item.className = 'preview-item';
                const canvas = document.createElement('canvas');
                this.loader.drawToCanvas(imgData, canvas, 4);
                const cap = document.createElement('div');
                cap.className = 'caption';
                cap.textContent = 'Label: ' + lblArr[i];
                item.appendChild(canvas);
                item.appendChild(cap);
                rowEl.appendChild(item);
            });

            block.appendChild(rowEl);
            container.appendChild(block);
        });
    }

    // ── Save / Load ───────────────────────────────────────────────────────────

    async onSave() {
        if (!this.aeMax && !this.aeAvg && !this.classifier)
            return this.log('ERROR: No model to save.');
        try {
            if (this.classifier) { await this.classifier.save('downloads://mnist-classifier'); this.log('Classifier saved.'); }
            if (this.aeMax)      { await this.aeMax.save('downloads://mnist-ae-max');          this.log('Max AE saved.'); }
            if (this.aeAvg)      { await this.aeAvg.save('downloads://mnist-ae-avg');          this.log('Avg AE saved.'); }
        } catch (err) { this.log('ERROR: ' + err.message); }
    }

    async onLoadModel() {
        const jsonF = document.getElementById('modelJsonFile').files[0];
        const binF  = document.getElementById('modelWeightsFile').files[0];
        if (!jsonF || !binF) return this.log('ERROR: Select .json and .bin files.');
        try {
            this.log('Loading model…');
            if (this.aeMax) this.aeMax.dispose();
            this.aeMax = await tf.loadLayersModel(tf.io.browserFiles([jsonF, binF]));
            this.updateModelInfo();
            this.log('Model loaded as Max-Pooling AE.');
        } catch (err) { this.log('ERROR: ' + err.message); }
    }

    onReset() {
        if (this.classifier) { this.classifier.dispose(); this.classifier = null; }
        if (this.aeMax)      { this.aeMax.dispose();      this.aeMax      = null; }
        if (this.aeAvg)      { this.aeAvg.dispose();      this.aeAvg      = null; }
        this.loader.dispose();
        this.trainData = null; this.testData = null;
        document.getElementById('dataStatus').textContent     = 'No data loaded';
        document.getElementById('modelInfo').textContent      = 'No model';
        document.getElementById('previewContainer').innerHTML = '';
        this.log('Reset done.');
    }

    // ── Model builders ────────────────────────────────────────────────────────

    buildClassifier() {
        const m = tf.sequential();
        m.add(tf.layers.conv2d({ inputShape:[28,28,1], filters:32, kernelSize:3, activation:'relu', padding:'same' }));
        m.add(tf.layers.conv2d({ filters:64, kernelSize:3, activation:'relu', padding:'same' }));
        m.add(tf.layers.maxPooling2d({ poolSize:2 }));
        m.add(tf.layers.dropout({ rate:0.25 }));
        m.add(tf.layers.flatten());
        m.add(tf.layers.dense({ units:128, activation:'relu' }));
        m.add(tf.layers.dropout({ rate:0.5 }));
        m.add(tf.layers.dense({ units:10, activation:'softmax' }));
        m.compile({ optimizer:'adam', loss:'categoricalCrossentropy', metrics:['accuracy'] });
        return m;
    }

    buildAutoencoder(poolType) {
        const model = tf.sequential();
        // Encoder
        model.add(tf.layers.conv2d({ inputShape:[28,28,1], filters:32, kernelSize:3, padding:'same', activation:'relu' }));
        if (poolType === 'max') {
            model.add(tf.layers.maxPooling2d({ poolSize:2 }));
        } else {
            model.add(tf.layers.averagePooling2d({ poolSize:2 }));
        }
        model.add(tf.layers.conv2d({ filters:64, kernelSize:3, padding:'same', activation:'relu' }));
        if (poolType === 'max') {
            model.add(tf.layers.maxPooling2d({ poolSize:2 }));
        } else {
            model.add(tf.layers.averagePooling2d({ poolSize:2 }));
        }
        // Decoder via Dense (avoids conv2dTranspose WebGL bugs)
        model.add(tf.layers.flatten());
        model.add(tf.layers.dense({ units:128, activation:'relu' }));
        model.add(tf.layers.dense({ units:784, activation:'sigmoid' }));
        model.add(tf.layers.reshape({ targetShape:[28,28,1] }));
        model.compile({ optimizer:'adam', loss:'meanSquaredError' });
        return model;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    updateModelInfo() {
        const p = (m) => { 
            if (!m) return 0;
            let n = 0; 
            m.layers.forEach(l => {
                try {
                    l.getWeights().forEach(w => n += w.size);
                } catch(e) {}
            }); 
            return n; 
        };
        let html = '';
        if (this.classifier) html += 'Classifier: '  + p(this.classifier).toLocaleString() + ' params<br>';
        if (this.aeMax)      html += 'AE Max-Pool: ' + p(this.aeMax).toLocaleString()      + ' params<br>';
        if (this.aeAvg)      html += 'AE Avg-Pool: ' + p(this.aeAvg).toLocaleString()      + ' params<br>';
        document.getElementById('modelInfo').innerHTML = html || 'No model';
    }

    log(msg) {
        const el   = document.getElementById('logs');
        const line = document.createElement('div');
        line.textContent = '[' + new Date().toLocaleTimeString() + '] ' + msg;
        if (msg.startsWith('ERROR')) line.style.color = '#ff6b6b';
        el.appendChild(line);
        el.scrollTop = el.scrollHeight;
    }
}

document.addEventListener('DOMContentLoaded', () => { new MNISTApp(); });
