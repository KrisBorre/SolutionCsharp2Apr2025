using Seq2SeqSharp;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Enums;
using Seq2SeqSharp.LearningRate;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;

// ChatGPT, modified
// Application uses 500 MegaByte of RAM. 
// 1 layer
namespace ConsoleDutchItalianSeq2Seq16Jul2025
{
    internal class Program
    {
        static void Main(string[] args)
        {
            int[] maxEpochNumbers = new int[]
            {
                3, 12, 20, 55, 67, 88, 110, 200, 255, 367, 488, 555
            };

            foreach (var maxEpochNum in maxEpochNumbers)
            {
                string srcLang = "NL";
                string tgtLang = "IT";
                string modelFilePath = "nl2it.model";

                // 70 sentence-pairs
                var trainData = new List<(string src, string tgt)>
                {
                    ("De hond slaapt.", "Il cane dorme."),
                    ("De hond rent.", "Il cane corre."),
                    ("De hond is groot.", "Il cane è grande."),
                    ("De hond is buiten.", "Il cane è fuori."),
                    ("De hond eet.", "Il cane mangia."),
                    ("De hond is moe.", "Il cane è stanco."),
                    ("De hond ligt op de bank.", "Il cane è sul divano."),
                    ("De hond zit op de vloer.", "Il cane è sul pavimento."),
                    ("De hond wacht.", "Il cane aspetta."),
                    ("De hond is blij.", "Il cane è felice."),

                    ("De kat ligt op de stoel.", "Il gatto è sulla sedia."),
                    ("De kat eet vis.", "Il gatto mangia pesce."),
                    ("De kat is wit.", "Il gatto è bianco."),
                    ("De kat speelt met een bal.", "Il gatto gioca con una palla."),
                    ("De kat zit op de tafel.", "Il gatto è sul tavolo."),
                    ("De kat is stil.", "Il gatto è tranquillo."),
                    ("De kat loopt weg.", "Il gatto se ne va."),
                    ("De kat miauwt.", "Il gatto miagola."),
                    ("De kat ligt in de zon.", "Il gatto è al sole."),
                    ("De kat kijkt naar de vogel.", "Il gatto guarda l’uccello."),

                    ("Ik zie de hond.", "Vedo il cane."),
                    ("Jij hebt een hond.", "Hai un cane."),
                    ("Zij houdt van haar hond.", "Lei ama il suo cane."),
                    ("Wij hebben twee katten.", "Abbiamo due gatti."),
                    ("Hij speelt met de kat.", "Lui gioca con il gatto."),
                    ("De hond is van mij.", "Il cane è mio."),
                    ("De kat is van haar.", "Il gatto è suo."),
                    ("Mijn hond is oud.", "Il mio cane è vecchio."),
                    ("Jouw kat is mooi.", "Il tuo gatto è bello."),
                    ("Hun hond is stil.", "Il loro cane è tranquillo."),

                    ("Ik eet brood.", "Mangio pane."),
                    ("Hij drinkt water.", "Lui beve acqua."),
                    ("Wij lezen een boek.", "Leggiamo un libro."),
                    ("Jij schrijft een brief.", "Scrivi una lettera."),
                    ("Zij kijkt naar de film.", "Lei guarda il film."),
                    ("Ik slaap op de bank.", "Dormo sul divano."),
                    ("Zij zit op de stoel.", "Lei è sulla sedia."),
                    ("Hij loopt naar school.", "Lui va a scuola."),
                    ("Wij spelen in het park.", "Giochiamo al parco."),
                    ("Jij zingt een lied.", "Canti una canzone."),

                    ("De hond is in de tuin.", "Il cane è in giardino."),
                    ("De kat is op het bed.", "Il gatto è sul letto."),
                    ("De hond gaat naar het park.", "Il cane va al parco."),
                    ("De kat blijft thuis.", "Il gatto resta a casa."),
                    ("De hond ligt onder de tafel.", "Il cane è sotto il tavolo."),
                    ("De kat slaapt op de vensterbank.", "Il gatto dorme sul davanzale."),
                    ("De hond is in de keuken.", "Il cane è in cucina."),
                    ("De kat zit in de doos.", "Il gatto è nella scatola."),
                    ("De hond is in de auto.", "Il cane è in macchina."),
                    ("De kat is op de trap.", "Il gatto è sulle scale."),

                    ("Waar is de hond?", "Dov'è il cane?"),
                    ("Waar slaapt de kat?", "Dove dorme il gatto?"),
                    ("Wat doet de hond?", "Cosa fa il cane?"),
                    ("Wie heeft een kat?", "Chi ha un gatto?"),
                    ("Waarom blaft de hond?", "Perché abbaia il cane?"),
                    ("Wanneer eet de kat?", "Quando mangia il gatto?"),
                    ("Hoe heet jouw hond?", "Come si chiama il tuo cane?"),
                    ("Is dat jouw kat?", "È il tuo gatto?"),
                    ("Wil je een hond?", "Vuoi un cane?"),
                    ("Heeft hij een kat?", "Ha un gatto?"),

                    ("Mijn hond is vriendelijk.", "Il mio cane è amichevole."),
                    ("Haar kat is nieuwsgierig.", "Il suo gatto è curioso."),
                    ("Hun hond is speels.", "Il loro cane è giocoso."),
                    ("Onze kat is oud.", "Il nostro gatto è vecchio."),
                    ("De zwarte kat slaapt.", "Il gatto nero dorme."),
                    ("De kleine hond blaft.", "Il piccolo cane abbaia."),
                    ("De hond en de kat slapen.", "Il cane e il gatto dormono."),
                    ("De hond speelt met de kat.", "Il cane gioca con il gatto."),
                    ("De kat zit naast de hond.", "Il gatto è accanto al cane."),
                    ("De hond kijkt naar de kat.", "Il cane guarda il gatto.")
                };

                string srcTrainFile = "train.nl.snt"; // Do not change file extension.
                string tgtTrainFile = "train.it.snt";

                File.WriteAllLines(srcTrainFile, trainData.ConvertAll(p => p.src));
                File.WriteAllLines(tgtTrainFile, trainData.ConvertAll(p => p.tgt));

                string rootPath = Directory.GetCurrentDirectory();

                var opts = new Seq2SeqOptions
                {
                    Task = ModeEnums.Train,
                    SrcLang = srcLang,
                    TgtLang = tgtLang,
                    EncoderLayerDepth = 1,
                    DecoderLayerDepth = 1,
                    HiddenSize = 1024,
                    SrcEmbeddingDim = 1024,
                    TgtEmbeddingDim = 1024,
                    MaxEpochNum = maxEpochNum,
                    MaxTokenSizePerBatch = 10,
                    ValMaxTokenSizePerBatch = 10,
                    MaxSrcSentLength = 20,
                    MaxTgtSentLength = 20,
                    PaddingType = PaddingEnums.AllowPadding,
                    TooLongSequence = TooLongSequence.Ignore,
                    ProcessorType = ProcessorTypeEnums.CPU,
                    ModelFilePath = modelFilePath,
                    StartLearningRate = 0.001f,
                    WarmUpSteps = 10,
                    SharedEmbeddings = false,
                    EncoderType = EncoderTypeEnums.BiLSTM,
                    DecoderType = DecoderTypeEnums.AttentionLSTM,
                    TrainCorpusPath = rootPath
                };

                var trainCorpus = new Seq2SeqCorpus(
                    corpusFilePath: opts.TrainCorpusPath,
                    srcLangName: srcLang,
                    tgtLangName: tgtLang,
                    maxTokenSizePerBatch: opts.MaxTokenSizePerBatch,
                    maxSrcSentLength: opts.MaxSrcSentLength,
                    maxTgtSentLength: opts.MaxTgtSentLength,
                    paddingEnums: opts.PaddingType,
                    tooLongSequence: opts.TooLongSequence);

                var (srcVocab, tgtVocab) = trainCorpus.BuildVocabs(1000, 1000, false);
                var learningRate = new DecayLearningRate(opts.StartLearningRate, opts.WarmUpSteps, 0, 0.7f, 100);
                var optimizer = Misc.CreateOptimizer(opts);
                var metrics = new List<IMetric> { new BleuMetric() };

                var model = new Seq2Seq(opts, srcVocab, tgtVocab);
                model.StatusUpdateWatcher += (s, e) =>
                {
                    if (e is CostEventArg cost)
                    {
                        Console.WriteLine($"Epoch {cost.Epoch}, Update {cost.Update}, Cost = {cost.AvgCostInTotal:F4}");
                    }
                };

                model.Train(
                    maxTrainingEpoch: opts.MaxEpochNum,
                    trainCorpus: trainCorpus,
                    validCorpusList: Array.Empty<Seq2SeqCorpus>(),
                    learningRate: learningRate,
                    optimizer: optimizer,
                    metrics: metrics.ToArray(),
                    decodingOptions: opts.CreateDecodingOptions());

                model.SaveModel(suffix: ".trained");

                // Inference
                opts.Task = ModeEnums.Test;
                opts.ModelFilePath = modelFilePath + ".trained";
                var inferModel = new Seq2Seq(opts);

                string testInputPath = "test_input.nl.snt";
                string testOutputPath = "test_output.it.snt";
                File.WriteAllLines(testInputPath, new[]
                {
                    "De hond zit in de tuin.",
                    "De kat eet op de tafel.",
                    "Mijn kat slaapt op het bed.",
                    "De kat kijkt naar de hond.",
                    "Waar is mijn kat?",
                    "Zij heeft een hond."
                 });

                inferModel.Test(
                    inputTestFile: testInputPath,
                    outputFile: testOutputPath,
                    batchSize: 1,
                    decodingOptions: opts.CreateDecodingOptions(),
                    srcSpmPath: null,
                    tgtSpmPath: null); // We are not using SentencePiece

                Console.WriteLine("Translations:");
                foreach (var line in File.ReadLines(testOutputPath))
                {
                    Console.WriteLine(line);
                }

                string[] files1 = Directory.GetFiles(rootPath, "*.tmp.sorted.txt");

                foreach (string file in files1)
                {
                    File.Delete(file);
                }

                string[] files2 = Directory.GetFiles(rootPath, "nl2it.model.*");
                foreach (string file in files2)
                {
                    File.Delete(file);
                }

                File.Delete(srcTrainFile);
                File.Delete(tgtTrainFile);
                File.Delete(testInputPath);
                File.Delete(testOutputPath);
            }

            /*
Epoch 2, Update 100, Cost = 8,4571
Translations:
<s> Il cane è il </s>
<s> Il gatto è sul </s>
<s> Il gatto è sul </s>
<s> Lei un il il </s>
<s> Il gatto è è </s>
<s> Lei un un </s>
Epoch 2, Update 100, Cost = 8,9963
Epoch 4, Update 200, Cost = 7,9738
Epoch 6, Update 300, Cost = 6,3352
Epoch 9, Update 400, Cost = 2,8276
Epoch 11, Update 500, Cost = 1,9485
Translations:
<s> Il cane è in giardino. </s>
<s> Il gatto è sul scale. </s>
<s> Il gatto dorme sul davanzale. </s>
<s> Il gatto guarda l'uccello. </s>
<s> Dov'è un il </s>
<s> Chi ha un gatto? </s>
Epoch 2, Update 100, Cost = 10,9080
Epoch 4, Update 200, Cost = 9,0712
Epoch 6, Update 300, Cost = 7,8617
Epoch 9, Update 400, Cost = 3,1929
Epoch 11, Update 500, Cost = 2,4564
Epoch 13, Update 600, Cost = 1,5906
Epoch 15, Update 700, Cost = 1,1403
Epoch 18, Update 800, Cost = 0,8296
Translations:
<s> Il cane è in pavimento. </s>
<s> Il gatto è sul </s>
<s> Il mio mio sul sul </s>
<s> Il gatto guarda l'uccello. </s>
<s> Dov'è il cane? </s>
<s> Lei guarda il gatto? </s>
Epoch 2, Update 100, Cost = 10,1403
Epoch 4, Update 200, Cost = 9,3775
Epoch 6, Update 300, Cost = 5,3977
Epoch 9, Update 400, Cost = 1,9281
Epoch 11, Update 500, Cost = 2,0555
Epoch 13, Update 600, Cost = 1,3446
Epoch 15, Update 700, Cost = 1,0242
Epoch 18, Update 800, Cost = 0,4850
Epoch 20, Update 900, Cost = 0,5839
Epoch 22, Update 1000, Cost = 0,5281
Epoch 24, Update 1100, Cost = 0,4926
Epoch 27, Update 1200, Cost = 0,3594
Epoch 29, Update 1300, Cost = 0,4025
Epoch 31, Update 1400, Cost = 0,4350
Epoch 34, Update 1500, Cost = 0,2208
Epoch 36, Update 1600, Cost = 0,4096
Epoch 38, Update 1700, Cost = 0,3718
Epoch 40, Update 1800, Cost = 0,3904
Epoch 43, Update 1900, Cost = 0,2434
Epoch 45, Update 2000, Cost = 0,3663
Epoch 47, Update 2100, Cost = 0,3739
Epoch 49, Update 2200, Cost = 0,3868
Epoch 52, Update 2300, Cost = 0,3036
Epoch 54, Update 2400, Cost = 0,3620
Translations:
<s> Il cane è in giardino. </s>
<s> Il gatto è sul </s>
<s> Il gatto dorme sul davanzale. </s>
<s> Il gatto guarda l'uccello. </s>
<s> Dov'è il cane? </s>
<s> Lei ama il gatto? </s>
Epoch 2, Update 100, Cost = 11,6657
Epoch 4, Update 200, Cost = 11,1330
Epoch 6, Update 300, Cost = 9,7550
Epoch 9, Update 400, Cost = 5,3247
Epoch 11, Update 500, Cost = 5,3270
Epoch 13, Update 600, Cost = 4,7553
Epoch 15, Update 700, Cost = 3,9115
Epoch 18, Update 800, Cost = 3,3635
Epoch 20, Update 900, Cost = 2,8897
Epoch 22, Update 1000, Cost = 3,1159
Epoch 24, Update 1100, Cost = 2,7596
Epoch 27, Update 1200, Cost = 2,9579
Epoch 29, Update 1300, Cost = 2,1455
Epoch 31, Update 1400, Cost = 2,5496
Epoch 34, Update 1500, Cost = 2,3021
Epoch 36, Update 1600, Cost = 2,5310
Epoch 38, Update 1700, Cost = 2,3927
Epoch 40, Update 1800, Cost = 2,3749
Epoch 43, Update 1900, Cost = 2,6157
Epoch 45, Update 2000, Cost = 2,3048
Epoch 47, Update 2100, Cost = 2,6547
Epoch 49, Update 2200, Cost = 2,4735
Epoch 52, Update 2300, Cost = 2,7684
Epoch 54, Update 2400, Cost = 2,0417
Epoch 56, Update 2500, Cost = 2,4640
Epoch 59, Update 2600, Cost = 2,2627
Epoch 61, Update 2700, Cost = 2,4928
Epoch 63, Update 2800, Cost = 2,3658
Epoch 65, Update 2900, Cost = 2,3567
Translations:
<s> Il cane è sul il tavolo. </s>
<s> Il gatto è sul tavolo. </s>
<s> Il gatto è sul vecchio. </s>
<s> Il gatto guarda l'uccello. </s>
<s> Dov'è il cane? </s>
<s> Lei un cane. </s>
Epoch 2, Update 100, Cost = 10,3999
Epoch 4, Update 200, Cost = 7,8509
Epoch 6, Update 300, Cost = 5,5425
Epoch 9, Update 400, Cost = 3,7315
Epoch 11, Update 500, Cost = 1,9836
Epoch 13, Update 600, Cost = 1,3078
Epoch 15, Update 700, Cost = 1,1427
Epoch 18, Update 800, Cost = 1,0844
Epoch 20, Update 900, Cost = 0,8105
Epoch 22, Update 1000, Cost = 0,7357
Epoch 24, Update 1100, Cost = 0,7262
Epoch 27, Update 1200, Cost = 0,7122
Epoch 29, Update 1300, Cost = 0,6009
Epoch 31, Update 1400, Cost = 0,6612
Epoch 34, Update 1500, Cost = 0,9068
Epoch 36, Update 1600, Cost = 0,6202
Epoch 38, Update 1700, Cost = 0,5442
Epoch 40, Update 1800, Cost = 0,6321
Epoch 43, Update 1900, Cost = 0,7049
Epoch 45, Update 2000, Cost = 0,5975
Epoch 47, Update 2100, Cost = 0,6041
Epoch 49, Update 2200, Cost = 0,6304
Epoch 52, Update 2300, Cost = 0,6410
Epoch 54, Update 2400, Cost = 0,5596
Epoch 56, Update 2500, Cost = 0,6334
Epoch 59, Update 2600, Cost = 0,8790
Epoch 61, Update 2700, Cost = 0,6065
Epoch 63, Update 2800, Cost = 0,5359
Epoch 65, Update 2900, Cost = 0,6263
Epoch 68, Update 3000, Cost = 0,6992
Epoch 70, Update 3100, Cost = 0,5945
Epoch 72, Update 3200, Cost = 0,6023
Epoch 74, Update 3300, Cost = 0,6291
Epoch 77, Update 3400, Cost = 0,6401
Epoch 79, Update 3500, Cost = 0,5590
Epoch 81, Update 3600, Cost = 0,6330
Epoch 84, Update 3700, Cost = 0,8786
Epoch 86, Update 3800, Cost = 0,6063
Translations:
<s> Il cane è in pavimento. </s>
<s> Il gatto mangia pesce. </s>
<s> Il gatto dorme sul davanzale. </s>
<s> Il gatto guarda l'uccello. </s>
<s> Dov'è il cane? </s>
<s> Chi un un </s>
Epoch 2, Update 100, Cost = 11,2397
Epoch 4, Update 200, Cost = 8,3885
Epoch 6, Update 300, Cost = 7,0437
Epoch 9, Update 400, Cost = 3,8027
Epoch 11, Update 500, Cost = 2,4323
Epoch 13, Update 600, Cost = 1,2740
Epoch 15, Update 700, Cost = 1,1190
Epoch 18, Update 800, Cost = 0,6944
Epoch 20, Update 900, Cost = 0,5997
Epoch 22, Update 1000, Cost = 0,5878
Epoch 24, Update 1100, Cost = 0,6214
Epoch 27, Update 1200, Cost = 0,4132
Epoch 29, Update 1300, Cost = 0,4046
Epoch 31, Update 1400, Cost = 0,5471
Epoch 34, Update 1500, Cost = 0,3910
Epoch 36, Update 1600, Cost = 0,4079
Epoch 38, Update 1700, Cost = 0,3757
Epoch 40, Update 1800, Cost = 0,5049
Epoch 43, Update 1900, Cost = 0,3277
Epoch 45, Update 2000, Cost = 0,3820
Epoch 47, Update 2100, Cost = 0,4368
Epoch 49, Update 2200, Cost = 0,5089
Epoch 52, Update 2300, Cost = 0,3483
Epoch 54, Update 2400, Cost = 0,3642
Epoch 56, Update 2500, Cost = 0,5110
Epoch 59, Update 2600, Cost = 0,3619
Epoch 61, Update 2700, Cost = 0,3936
Epoch 63, Update 2800, Cost = 0,3671
Epoch 65, Update 2900, Cost = 0,4970
Epoch 68, Update 3000, Cost = 0,3227
Epoch 70, Update 3100, Cost = 0,3789
Epoch 72, Update 3200, Cost = 0,4348
Epoch 74, Update 3300, Cost = 0,5072
Epoch 77, Update 3400, Cost = 0,3473
Epoch 79, Update 3500, Cost = 0,3636
Epoch 81, Update 3600, Cost = 0,5106
Epoch 84, Update 3700, Cost = 0,3616
Epoch 86, Update 3800, Cost = 0,3935
Epoch 88, Update 3900, Cost = 0,3670
Epoch 90, Update 4000, Cost = 0,4970
Epoch 93, Update 4100, Cost = 0,3227
Epoch 95, Update 4200, Cost = 0,3789
Epoch 97, Update 4300, Cost = 0,4348
Epoch 99, Update 4400, Cost = 0,5072
Epoch 102, Update 4500, Cost = 0,3473
Epoch 104, Update 4600, Cost = 0,3636
Epoch 106, Update 4700, Cost = 0,5106
Epoch 109, Update 4800, Cost = 0,3616
Translations:
<s> Il cane è sul pavimento. </s>
<s> Il gatto è pesce. </s>
<s> Il gatto dorme sul davanzale. </s>
<s> Il gatto guarda l'uccello. </s>
<s> Dov'è il cane? </s>
<s> Lei guarda il cane. </s>
Epoch 2, Update 100, Cost = 11,7823
Epoch 4, Update 200, Cost = 12,3459
Epoch 6, Update 300, Cost = 9,7519
Epoch 9, Update 400, Cost = 3,7053
Epoch 11, Update 500, Cost = 4,3075
Epoch 13, Update 600, Cost = 3,7771
Epoch 15, Update 700, Cost = 3,0278
Epoch 18, Update 800, Cost = 1,4794
Epoch 20, Update 900, Cost = 2,0059
Epoch 22, Update 1000, Cost = 1,8458
Epoch 24, Update 1100, Cost = 1,9621
Epoch 27, Update 1200, Cost = 1,0895
Epoch 29, Update 1300, Cost = 1,7642
Epoch 31, Update 1400, Cost = 1,8614
Epoch 34, Update 1500, Cost = 1,0640
Epoch 36, Update 1600, Cost = 1,4540
Epoch 38, Update 1700, Cost = 1,6130
Epoch 40, Update 1800, Cost = 1,7725
Epoch 43, Update 1900, Cost = 0,8983
Epoch 45, Update 2000, Cost = 1,5191
Epoch 47, Update 2100, Cost = 1,5241
Epoch 49, Update 2200, Cost = 1,7154
Epoch 52, Update 2300, Cost = 0,9602
Epoch 54, Update 2400, Cost = 1,6632
Epoch 56, Update 2500, Cost = 1,7866
Epoch 59, Update 2600, Cost = 1,0235
Epoch 61, Update 2700, Cost = 1,4185
Epoch 63, Update 2800, Cost = 1,5894
Epoch 65, Update 2900, Cost = 1,7563
Epoch 68, Update 3000, Cost = 0,8900
Epoch 70, Update 3100, Cost = 1,5116
Epoch 72, Update 3200, Cost = 1,5191
Epoch 74, Update 3300, Cost = 1,7119
Epoch 77, Update 3400, Cost = 0,9583
Epoch 79, Update 3500, Cost = 1,6617
Epoch 81, Update 3600, Cost = 1,7856
Epoch 84, Update 3700, Cost = 1,0229
Epoch 86, Update 3800, Cost = 1,4182
Epoch 88, Update 3900, Cost = 1,5892
Epoch 90, Update 4000, Cost = 1,7562
Epoch 93, Update 4100, Cost = 0,8900
Epoch 95, Update 4200, Cost = 1,5116
Epoch 97, Update 4300, Cost = 1,5190
Epoch 99, Update 4400, Cost = 1,7118
Epoch 102, Update 4500, Cost = 0,9583
Epoch 104, Update 4600, Cost = 1,6617
Epoch 106, Update 4700, Cost = 1,7856
Epoch 109, Update 4800, Cost = 1,0229
Epoch 111, Update 4900, Cost = 1,4182
Epoch 113, Update 5000, Cost = 1,5892
Epoch 115, Update 5100, Cost = 1,7562
Epoch 118, Update 5200, Cost = 0,8900
Epoch 120, Update 5300, Cost = 1,5116
Epoch 122, Update 5400, Cost = 1,5190
Epoch 124, Update 5500, Cost = 1,7118
Epoch 127, Update 5600, Cost = 0,9583
Epoch 129, Update 5700, Cost = 1,6617
Epoch 131, Update 5800, Cost = 1,7856
Epoch 134, Update 5900, Cost = 1,0229
Epoch 136, Update 6000, Cost = 1,4182
Epoch 138, Update 6100, Cost = 1,5892
Epoch 140, Update 6200, Cost = 1,7562
Epoch 143, Update 6300, Cost = 0,8900
Epoch 145, Update 6400, Cost = 1,5116
Epoch 147, Update 6500, Cost = 1,5190
Epoch 149, Update 6600, Cost = 1,7118
Epoch 152, Update 6700, Cost = 0,9583
Epoch 154, Update 6800, Cost = 1,6617
Epoch 156, Update 6900, Cost = 1,7856
Epoch 159, Update 7000, Cost = 1,0229
Epoch 161, Update 7100, Cost = 1,4182
Epoch 163, Update 7200, Cost = 1,5892
Epoch 165, Update 7300, Cost = 1,7562
Epoch 168, Update 7400, Cost = 0,8900
Epoch 170, Update 7500, Cost = 1,5116
Epoch 172, Update 7600, Cost = 1,5190
Epoch 174, Update 7700, Cost = 1,7118
Epoch 177, Update 7800, Cost = 0,9583
Epoch 179, Update 7900, Cost = 1,6617
Epoch 181, Update 8000, Cost = 1,7856
Epoch 184, Update 8100, Cost = 1,0229
Epoch 186, Update 8200, Cost = 1,4182
Epoch 188, Update 8300, Cost = 1,5892
Epoch 190, Update 8400, Cost = 1,7562
Epoch 193, Update 8500, Cost = 0,8900
Epoch 195, Update 8600, Cost = 1,5116
Epoch 197, Update 8700, Cost = 1,5190
Epoch 199, Update 8800, Cost = 1,7118
Translations:
<s> Il cane è in giardino. </s>
<s> Il gatto è sul </s>
<s> Il gatto è sul davanzale. </s>
<s> Il gatto guarda l'uccello. </s>
<s> Il loro cane gatto? </s>
<s> Lei un un </s>
Epoch 2, Update 100, Cost = 12,1011
Epoch 4, Update 200, Cost = 11,0120
Epoch 6, Update 300, Cost = 6,3865
Epoch 9, Update 400, Cost = 1,6836
Epoch 11, Update 500, Cost = 1,9225
Epoch 13, Update 600, Cost = 1,1092
Epoch 15, Update 700, Cost = 0,8788
Epoch 18, Update 800, Cost = 0,2485
Epoch 20, Update 900, Cost = 0,4218
Epoch 22, Update 1000, Cost = 0,3562
Epoch 24, Update 1100, Cost = 0,4044
           */

            /*
            Epoch 2, Update 100, Cost = 7,6434
Translations:
<s> Il cane il il </s>
<s> Il gatto il il </s>
<s> Il gatto è </s>
<s> Il gatto il il il </s>
<s> Ha gatto? </s>
<s> una una una una una una una canzone. una canzone. canzone. canzone. </s>
Epoch 2, Update 100, Cost = 9,3758
Epoch 4, Update 200, Cost = 7,6376
Epoch 6, Update 300, Cost = 12,1833
Epoch 9, Update 400, Cost = 3,5533
Epoch 11, Update 500, Cost = 2,6668
Translations:
<s> Il cane è in il </s>
<s> Dormo dorme il cane? cane? gatto è cane? cane? cane? gatto è cane? cane? cane? gatto è cane? cane? cane? gatto è cane? cane? cane? gatto è cane? cane? cane? gatto è cane? cane? cane? gatto è cane? cane? cane? gatto è cane? cane? cane? gatto è cane? cane? cane? gatto è cane? cane? cane? gatto è cane? cane? cane? gatto è cane?
<s> Il gatto dorme sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul sul
<s> Il gatto guarda il accanto il cane. il accanto il accanto il accanto il accanto il accanto il accanto il accanto il cane? il cane? il cane? il cane? il accanto il cane? il accanto il cane? il accanto il cane? il accanto il cane? il accanto il cane? il accanto il cane? il cane? il accanto il cane? il accanto il cane?
<s> Il nostro gatto è vecchio. </s>
<s> Lei un il gatto? il tuo gatto? </s>
Epoch 2, Update 100, Cost = 9,8069
Epoch 4, Update 200, Cost = 8,4937
Epoch 6, Update 300, Cost = 5,0302
Epoch 9, Update 400, Cost = 0,6096
Epoch 11, Update 500, Cost = 0,7463
Epoch 13, Update 600, Cost = 4,3793
Epoch 15, Update 700, Cost = 3,0378
Epoch 18, Update 800, Cost = 2,0270
Translations:
<s> Il cane è in giardino. </s>
<s> Il gatto è sul </s>
<s> Il gatto dorme sul davanzale. </s>
<s> Il gatto è cane. </s>
<s> Dov'è il cane? </s>
<s> Hai un cane. </s>
Epoch 2, Update 100, Cost = 9,8718
Epoch 4, Update 200, Cost = 6,8817
Epoch 6, Update 300, Cost = 4,3668
Epoch 9, Update 400, Cost = 1,1858
Epoch 11, Update 500, Cost = 5,9820
Epoch 13, Update 600, Cost = 3,3994
Epoch 15, Update 700, Cost = 2,2606
Epoch 18, Update 800, Cost = 1,6539
Epoch 20, Update 900, Cost = 1,7298
Epoch 22, Update 1000, Cost = 1,4552
Epoch 24, Update 1100, Cost = 1,2338
Epoch 27, Update 1200, Cost = 1,1131
Epoch 29, Update 1300, Cost = 1,1674
Epoch 31, Update 1400, Cost = 1,1285
Epoch 34, Update 1500, Cost = 1,0204
Epoch 36, Update 1600, Cost = 1,1788
Epoch 38, Update 1700, Cost = 1,0732
Epoch 40, Update 1800, Cost = 1,0157
Epoch 43, Update 1900, Cost = 0,9010
Epoch 45, Update 2000, Cost = 1,0819
Epoch 47, Update 2100, Cost = 1,0731
Epoch 49, Update 2200, Cost = 0,9809
Epoch 52, Update 2300, Cost = 0,9265
Epoch 54, Update 2400, Cost = 1,0647
Translations:
<s> Il cane è in giardino. </s>
<s> Il gatto è sul </s>
<s> Il mio gatto sul </s>
<s> Il gatto guarda l'uccello. </s>
<s> Dov'è il cane? </s>
<s> Lei un gatto? </s>
Epoch 2, Update 100, Cost = 8,9322
Epoch 4, Update 200, Cost = 5,6255
Epoch 6, Update 300, Cost = 4,3387
Epoch 9, Update 400, Cost = 5,3760
Epoch 11, Update 500, Cost = 4,4092
Epoch 13, Update 600, Cost = 2,8155
Epoch 15, Update 700, Cost = 2,1509
Epoch 18, Update 800, Cost = 1,7524
Epoch 20, Update 900, Cost = 1,4552
Epoch 22, Update 1000, Cost = 1,3377
Epoch 24, Update 1100, Cost = 1,2239
Epoch 27, Update 1200, Cost = 1,1022
Epoch 29, Update 1300, Cost = 1,0044
Epoch 31, Update 1400, Cost = 1,0825
Epoch 34, Update 1500, Cost = 0,8545
Epoch 36, Update 1600, Cost = 1,0743
Epoch 38, Update 1700, Cost = 0,9658
Epoch 40, Update 1800, Cost = 1,0250
Epoch 43, Update 1900, Cost = 1,0037
Epoch 45, Update 2000, Cost = 0,9895
Epoch 47, Update 2100, Cost = 1,0404
Epoch 49, Update 2200, Cost = 1,0197
Epoch 52, Update 2300, Cost = 0,9709
Epoch 54, Update 2400, Cost = 0,9267
Epoch 56, Update 2500, Cost = 1,0271
Epoch 59, Update 2600, Cost = 0,8216
Epoch 61, Update 2700, Cost = 1,0484
Epoch 63, Update 2800, Cost = 0,9493
Epoch 65, Update 2900, Cost = 1,0132
Translations:
<s> Il cane è in giardino. </s>
<s> Il gatto è sul tavolo. </s>
<s> Il gatto dorme sul davanzale. </s>
<s> Il gatto guarda l'uccello. </s>
<s> Dov'è il cane? </s>
<s> Chi un un gatto? </s>
Epoch 2, Update 100, Cost = 9,9073
Epoch 4, Update 200, Cost = 8,6941
Epoch 6, Update 300, Cost = 8,8265
Epoch 9, Update 400, Cost = 1,8754
Epoch 11, Update 500, Cost = 2,5164
Epoch 13, Update 600, Cost = 1,9441
Epoch 15, Update 700, Cost = 4,4521
Epoch 18, Update 800, Cost = 3,1107
Epoch 20, Update 900, Cost = 3,3395
Epoch 22, Update 1000, Cost = 2,8806
Epoch 24, Update 1100, Cost = 2,5025
Epoch 27, Update 1200, Cost = 2,3674
Epoch 29, Update 1300, Cost = 2,2717
Epoch 31, Update 1400, Cost = 2,3175
Epoch 34, Update 1500, Cost = 2,0208
Epoch 36, Update 1600, Cost = 2,5344
Epoch 38, Update 1700, Cost = 2,1504
Epoch 40, Update 1800, Cost = 2,0929
Epoch 43, Update 1900, Cost = 1,9270
Epoch 45, Update 2000, Cost = 2,2268
Epoch 47, Update 2100, Cost = 2,1221
Epoch 49, Update 2200, Cost = 2,0692
Epoch 52, Update 2300, Cost = 2,0079
Epoch 54, Update 2400, Cost = 2,0275
Epoch 56, Update 2500, Cost = 2,1433
Epoch 59, Update 2600, Cost = 1,9707
Epoch 61, Update 2700, Cost = 2,4483
Epoch 63, Update 2800, Cost = 2,0975
Epoch 65, Update 2900, Cost = 2,0545
Epoch 68, Update 3000, Cost = 1,9144
Epoch 70, Update 3100, Cost = 2,2102
Epoch 72, Update 3200, Cost = 2,1109
Epoch 74, Update 3300, Cost = 2,0608
Epoch 77, Update 3400, Cost = 2,0040
Epoch 79, Update 3500, Cost = 2,0242
Epoch 81, Update 3600, Cost = 2,1410
Epoch 84, Update 3700, Cost = 1,9701
Epoch 86, Update 3800, Cost = 2,4474
Translations:
<s> Il cane è in giardino. </s>
<s> Il gatto è sul tavolo. </s>
<s> Il gatto è sul </s>
<s> Il gatto guarda l'uccello. </s>
<s> Il il cane? </s>
<s> Lei guarda il </s>
Epoch 2, Update 100, Cost = 7,6574
Epoch 4, Update 200, Cost = 4,0885
Epoch 6, Update 300, Cost = 2,2249
Epoch 9, Update 400, Cost = 3,6764
Epoch 11, Update 500, Cost = 2,3033
Epoch 13, Update 600, Cost = 1,3053
Epoch 15, Update 700, Cost = 0,9622
Epoch 18, Update 800, Cost = 2,5989
Epoch 20, Update 900, Cost = 2,1935
Epoch 22, Update 1000, Cost = 1,8156
Epoch 24, Update 1100, Cost = 1,7081
Epoch 27, Update 1200, Cost = 1,2841
Epoch 29, Update 1300, Cost = 1,1444
Epoch 31, Update 1400, Cost = 1,1585
Epoch 34, Update 1500, Cost = 1,0410
Epoch 36, Update 1600, Cost = 1,2408
Epoch 38, Update 1700, Cost = 1,0440
Epoch 40, Update 1800, Cost = 1,0255
Epoch 43, Update 1900, Cost = 0,9633
Epoch 45, Update 2000, Cost = 1,0779
Epoch 47, Update 2100, Cost = 1,0366
Epoch 49, Update 2200, Cost = 1,1239
Epoch 52, Update 2300, Cost = 1,0332
Epoch 54, Update 2400, Cost = 0,9867
Epoch 56, Update 2500, Cost = 1,0343
Epoch 59, Update 2600, Cost = 0,9743
Epoch 61, Update 2700, Cost = 1,1887
Epoch 63, Update 2800, Cost = 1,0104
Epoch 65, Update 2900, Cost = 1,0008
Epoch 68, Update 3000, Cost = 0,9476
Epoch 70, Update 3100, Cost = 1,0673
Epoch 72, Update 3200, Cost = 1,0291
Epoch 74, Update 3300, Cost = 1,1176
Epoch 77, Update 3400, Cost = 1,0298
Epoch 79, Update 3500, Cost = 0,9846
Epoch 81, Update 3600, Cost = 1,0326
Epoch 84, Update 3700, Cost = 0,9728
Epoch 86, Update 3800, Cost = 1,1881
Epoch 88, Update 3900, Cost = 1,0101
Epoch 90, Update 4000, Cost = 1,0005
Epoch 93, Update 4100, Cost = 0,9474
Epoch 95, Update 4200, Cost = 1,0673
Epoch 97, Update 4300, Cost = 1,0291
Epoch 99, Update 4400, Cost = 1,1176
Epoch 102, Update 4500, Cost = 1,0298
Epoch 104, Update 4600, Cost = 0,9846
Epoch 106, Update 4700, Cost = 1,0326
Epoch 109, Update 4800, Cost = 0,9728
Translations:
<s> Il cane è in giardino. </s>
<s> Il gatto mangia sul </s>
<s> Il gatto dorme sul davanzale. </s>
<s> Vedo il cane. cane. </s>
<s> Dove un gatto? </s>
<s> Hai un cane. </s>
             */

            /*
             Epoch 2, Update 100, Cost = 10,6228
Translations:
<s> Il cane è </s>
<s> Il gatto è </s>
<s> Il gatto è è </s>
<s> Il gatto il </s>
<s> Il gatto è </s>
<s> Giochiamo un </s>
Epoch 2, Update 100, Cost = 9,2287
Epoch 4, Update 200, Cost = 6,3803
Epoch 6, Update 300, Cost = 12,5790
Epoch 9, Update 400, Cost = 2,5919
Epoch 11, Update 500, Cost = 2,9896
Translations:
<s> Il cane è in il </s>
<s> Il gatto è sul scale. </s>
<s> Il gatto è sul davanzale. </s>
<s> Il gatto guarda il al </s>
<s> Il il cane? </s>
<s> Vuoi un gatto? </s>
Epoch 2, Update 100, Cost = 11,0710
Epoch 4, Update 200, Cost = 12,3156
Epoch 6, Update 300, Cost = 7,2841
Epoch 9, Update 400, Cost = 1,6935
Epoch 11, Update 500, Cost = 1,6907
Epoch 13, Update 600, Cost = 0,9361
Epoch 15, Update 700, Cost = 3,4956
Epoch 18, Update 800, Cost = 1,9221
Translations:
<s> Il cane è in giardino. </s>
<s> Il gatto è sul tavolo. </s>
<s> Il gatto è sul letto. </s>
<s> Il gatto guarda l'uccello. </s>
<s> Dove un gatto? </s>
<s> Lei un gatto? </s>
Epoch 2, Update 100, Cost = 9,5177
Epoch 4, Update 200, Cost = 12,7092
Epoch 6, Update 300, Cost = 7,4815
Epoch 9, Update 400, Cost = 1,1999
Epoch 11, Update 500, Cost = 0,7052
Epoch 13, Update 600, Cost = 0,6088
Epoch 15, Update 700, Cost = 3,2366
Epoch 18, Update 800, Cost = 1,8435
Epoch 20, Update 900, Cost = 2,1380
Epoch 22, Update 1000, Cost = 1,6951
Epoch 24, Update 1100, Cost = 1,3975
Epoch 27, Update 1200, Cost = 0,9228
Epoch 29, Update 1300, Cost = 1,3493
Epoch 31, Update 1400, Cost = 1,2558
Epoch 34, Update 1500, Cost = 0,7185
Epoch 36, Update 1600, Cost = 1,3717
Epoch 38, Update 1700, Cost = 1,1510
Epoch 40, Update 1800, Cost = 1,1361
Epoch 43, Update 1900, Cost = 0,6949
Epoch 45, Update 2000, Cost = 1,2185
Epoch 47, Update 2100, Cost = 1,0736
Epoch 49, Update 2200, Cost = 1,0986
Epoch 52, Update 2300, Cost = 0,7360
Epoch 54, Update 2400, Cost = 1,1992
Translations:
<s> Il cane è in pavimento. </s>
<s> Il gatto è sul tavolo. </s>
<s> Il gatto è sul letto. </s>
<s> Il gatto guarda l'uccello. </s>
<s> Il nostro il gatto? </s>
<s> Lei un un </s>
Epoch 2, Update 100, Cost = 9,2636
Epoch 4, Update 200, Cost = 7,1365
Epoch 6, Update 300, Cost = 5,3108
Epoch 9, Update 400, Cost = 10,9437
Epoch 11, Update 500, Cost = 8,8596
Epoch 13, Update 600, Cost = 6,7066
Epoch 15, Update 700, Cost = 5,4750
Epoch 18, Update 800, Cost = 3,5985
Epoch 20, Update 900, Cost = 4,0201
Epoch 22, Update 1000, Cost = 3,4774
Epoch 24, Update 1100, Cost = 3,4916
Epoch 27, Update 1200, Cost = 2,8521
Epoch 29, Update 1300, Cost = 2,4893
Epoch 31, Update 1400, Cost = 3,0594
Epoch 34, Update 1500, Cost = 2,6677
Epoch 36, Update 1600, Cost = 2,3711
Epoch 38, Update 1700, Cost = 2,4351
Epoch 40, Update 1800, Cost = 2,9646
Epoch 43, Update 1900, Cost = 2,4038
Epoch 45, Update 2000, Cost = 2,3719
Epoch 47, Update 2100, Cost = 2,5782
Epoch 49, Update 2200, Cost = 2,9017
Epoch 52, Update 2300, Cost = 2,3760
Epoch 54, Update 2400, Cost = 2,2287
Epoch 56, Update 2500, Cost = 2,8926
Epoch 59, Update 2600, Cost = 2,5401
Epoch 61, Update 2700, Cost = 2,2677
Epoch 63, Update 2800, Cost = 2,3568
Epoch 65, Update 2900, Cost = 2,9339
Translations:
<s> Il gatto è sul </s>
<s> Il cane è sotto il </s>
<s> Il gatto è sul </s>
<s> Vedo il cane. </s>
<s> Il cane il </s>
<s> Lei un cane? </s>
Epoch 2, Update 100, Cost = 11,5516
Epoch 4, Update 200, Cost = 7,6155
Epoch 6, Update 300, Cost = 4,4378
Epoch 9, Update 400, Cost = 0,8253
Epoch 11, Update 500, Cost = 0,8406
Epoch 13, Update 600, Cost = 4,9204
Epoch 15, Update 700, Cost = 3,3688
Epoch 18, Update 800, Cost = 2,1434
Epoch 20, Update 900, Cost = 2,0893
Epoch 22, Update 1000, Cost = 1,6967
Epoch 24, Update 1100, Cost = 1,6349
Epoch 27, Update 1200, Cost = 1,2782
Epoch 29, Update 1300, Cost = 1,2995
Epoch 31, Update 1400, Cost = 1,4634
Epoch 34, Update 1500, Cost = 1,0119
Epoch 36, Update 1600, Cost = 1,4293
Epoch 38, Update 1700, Cost = 1,2303
Epoch 40, Update 1800, Cost = 1,3198
Epoch 43, Update 1900, Cost = 1,0095
Epoch 45, Update 2000, Cost = 1,2514
Epoch 47, Update 2100, Cost = 1,2345
Epoch 49, Update 2200, Cost = 1,3159
Epoch 52, Update 2300, Cost = 1,0861
Epoch 54, Update 2400, Cost = 1,1667
Epoch 56, Update 2500, Cost = 1,3441
Epoch 59, Update 2600, Cost = 0,9734
Epoch 61, Update 2700, Cost = 1,3874
Epoch 63, Update 2800, Cost = 1,2019
Epoch 65, Update 2900, Cost = 1,2966
Epoch 68, Update 3000, Cost = 0,9990
Epoch 70, Update 3100, Cost = 1,2432
Epoch 72, Update 3200, Cost = 1,2280
Epoch 74, Update 3300, Cost = 1,3109
Epoch 77, Update 3400, Cost = 1,0838
Epoch 79, Update 3500, Cost = 1,1652
Epoch 81, Update 3600, Cost = 1,3428
Epoch 84, Update 3700, Cost = 0,9727
Epoch 86, Update 3800, Cost = 1,3867
Translations:
<s> Il cane è in giardino. </s>
<s> Il gatto è sul tavolo. </s>
<s> Il gatto è sul letto. </s>
<s> Il gatto guarda l'uccello. </s>
<s> Il gatto è gatto? </s>
<s> Hai un cane. </s>
Epoch 2, Update 100, Cost = 9,3069
Epoch 4, Update 200, Cost = 14,7703
Epoch 6, Update 300, Cost = 9,5717
Epoch 9, Update 400, Cost = 1,3321
Epoch 11, Update 500, Cost = 2,8076
Epoch 13, Update 600, Cost = 1,5202
Epoch 15, Update 700, Cost = 3,4772
Epoch 18, Update 800, Cost = 2,0366
Epoch 20, Update 900, Cost = 2,0502
Epoch 22, Update 1000, Cost = 1,5966
Epoch 24, Update 1100, Cost = 1,4007
Epoch 27, Update 1200, Cost = 1,0906
Epoch 29, Update 1300, Cost = 1,1977
Epoch 31, Update 1400, Cost = 1,1845
Epoch 34, Update 1500, Cost = 0,9014
Epoch 36, Update 1600, Cost = 1,1557
Epoch 38, Update 1700, Cost = 1,0063
Epoch 40, Update 1800, Cost = 1,0456
Epoch 43, Update 1900, Cost = 0,7990
Epoch 45, Update 2000, Cost = 1,0812
Epoch 47, Update 2100, Cost = 0,9694
Epoch 49, Update 2200, Cost = 1,0099
Epoch 52, Update 2300, Cost = 0,8857
Epoch 54, Update 2400, Cost = 1,0117
Epoch 56, Update 2500, Cost = 1,0446
Epoch 59, Update 2600, Cost = 0,8502
Epoch 61, Update 2700, Cost = 1,1040
Epoch 63, Update 2800, Cost = 0,9739
Epoch 65, Update 2900, Cost = 1,0201
Epoch 68, Update 3000, Cost = 0,7840
Epoch 70, Update 3100, Cost = 1,0704
Epoch 72, Update 3200, Cost = 0,9625
Epoch 74, Update 3300, Cost = 1,0047
Epoch 77, Update 3400, Cost = 0,8820
Epoch 79, Update 3500, Cost = 1,0096
Epoch 81, Update 3600, Cost = 1,0431
Epoch 84, Update 3700, Cost = 0,8494
Epoch 86, Update 3800, Cost = 1,1030
Epoch 88, Update 3900, Cost = 0,9732
Epoch 90, Update 4000, Cost = 1,0199
Epoch 93, Update 4100, Cost = 0,7839
Epoch 95, Update 4200, Cost = 1,0703
Epoch 97, Update 4300, Cost = 0,9624
Epoch 99, Update 4400, Cost = 1,0047
Epoch 102, Update 4500, Cost = 0,8820
Epoch 104, Update 4600, Cost = 1,0095
Epoch 106, Update 4700, Cost = 1,0431
Epoch 109, Update 4800, Cost = 0,8494
Translations:
<s> Il cane è in giardino. </s>
<s> Il gatto è sul davanzale. </s>
<s> Il gatto è sul letto. </s>
<s> Il gatto guarda l'uccello. </s>
<s> È il gatto? </s>
<s> Chi un un </s>
Epoch 2, Update 100, Cost = 11,3188
Epoch 4, Update 200, Cost = 14,6935
Epoch 6, Update 300, Cost = 11,7812
Epoch 9, Update 400, Cost = 3,4475
Epoch 11, Update 500, Cost = 6,5270
Epoch 13, Update 600, Cost = 5,1854
Epoch 15, Update 700, Cost = 7,9057
Epoch 18, Update 800, Cost = 6,1983
Epoch 20, Update 900, Cost = 6,4135
Epoch 22, Update 1000, Cost = 6,4440
Epoch 24, Update 1100, Cost = 5,6290
Epoch 27, Update 1200, Cost = 5,5715
Epoch 29, Update 1300, Cost = 4,9988
Epoch 31, Update 1400, Cost = 5,3085
Epoch 34, Update 1500, Cost = 4,4429
Epoch 36, Update 1600, Cost = 5,6254
Epoch 38, Update 1700, Cost = 5,1864
Epoch 40, Update 1800, Cost = 4,8277
Epoch 43, Update 1900, Cost = 4,6476
Epoch 45, Update 2000, Cost = 5,1305
Epoch 47, Update 2100, Cost = 5,2540
Epoch 49, Update 2200, Cost = 4,9559
Epoch 52, Update 2300, Cost = 5,1848
Epoch 54, Update 2400, Cost = 4,6770
Epoch 56, Update 2500, Cost = 5,0457
Epoch 59, Update 2600, Cost = 4,3866
Epoch 61, Update 2700, Cost = 5,5471
Epoch 63, Update 2800, Cost = 5,1279
Epoch 65, Update 2900, Cost = 4,7831
Epoch 68, Update 3000, Cost = 4,6141
Epoch 70, Update 3100, Cost = 5,1142
Epoch 72, Update 3200, Cost = 5,2413
Epoch 74, Update 3300, Cost = 4,9466
Epoch 77, Update 3400, Cost = 5,1774
Epoch 79, Update 3500, Cost = 4,6738
Epoch 81, Update 3600, Cost = 5,0433
Epoch 84, Update 3700, Cost = 4,3850
Epoch 86, Update 3800, Cost = 5,5455
Epoch 88, Update 3900, Cost = 5,1267
Epoch 90, Update 4000, Cost = 4,7828
Epoch 93, Update 4100, Cost = 4,6139
Epoch 95, Update 4200, Cost = 5,1140
Epoch 97, Update 4300, Cost = 5,2411
Epoch 99, Update 4400, Cost = 4,9466
Epoch 102, Update 4500, Cost = 5,1774
Epoch 104, Update 4600, Cost = 4,6738
Epoch 106, Update 4700, Cost = 5,0433
Epoch 109, Update 4800, Cost = 4,3850
Epoch 111, Update 4900, Cost = 5,5455
Epoch 113, Update 5000, Cost = 5,1267
Epoch 115, Update 5100, Cost = 4,7828
Epoch 118, Update 5200, Cost = 4,6139
Epoch 120, Update 5300, Cost = 5,1140
Epoch 122, Update 5400, Cost = 5,2411
Epoch 124, Update 5500, Cost = 4,9466
Epoch 127, Update 5600, Cost = 5,1774
Epoch 129, Update 5700, Cost = 4,6738
Epoch 131, Update 5800, Cost = 5,0433
Epoch 134, Update 5900, Cost = 4,3850
Epoch 136, Update 6000, Cost = 5,5455
Epoch 138, Update 6100, Cost = 5,1267
Epoch 140, Update 6200, Cost = 4,7828
Epoch 143, Update 6300, Cost = 4,6139
Epoch 145, Update 6400, Cost = 5,1140
Epoch 147, Update 6500, Cost = 5,2411
Epoch 149, Update 6600, Cost = 4,9466
Epoch 152, Update 6700, Cost = 5,1774
Epoch 154, Update 6800, Cost = 4,6738
Epoch 156, Update 6900, Cost = 5,0433
Epoch 159, Update 7000, Cost = 4,3850
Epoch 161, Update 7100, Cost = 5,5455
Epoch 163, Update 7200, Cost = 5,1267
Epoch 165, Update 7300, Cost = 4,7828
Epoch 168, Update 7400, Cost = 4,6139
Epoch 170, Update 7500, Cost = 5,1140
Epoch 172, Update 7600, Cost = 5,2411
Epoch 174, Update 7700, Cost = 4,9466
Epoch 177, Update 7800, Cost = 5,1774
Epoch 179, Update 7900, Cost = 4,6738
Epoch 181, Update 8000, Cost = 5,0433
Epoch 184, Update 8100, Cost = 4,3850
Epoch 186, Update 8200, Cost = 5,5455
Epoch 188, Update 8300, Cost = 5,1267
Epoch 190, Update 8400, Cost = 4,7828
Epoch 193, Update 8500, Cost = 4,6139
Epoch 195, Update 8600, Cost = 5,1140
Epoch 197, Update 8700, Cost = 5,2411
Epoch 199, Update 8800, Cost = 4,9466
Translations:
<s> Il cane è in </s>
<s> Il gatto è sul </s>
<s> Il gatto è sul </s>
<s> Il gatto è sul </s>
<s> Dov'è il cane? </s>
<s> Lei un il gatto? </s>
             */

            /*
             Epoch 2, Update 100, Cost = 8,9059
Translations:
<s> Il cane è è </s>
<s> Il gatto è sul </s>
<s> Il gatto è è </s>
<s> Il gatto gatto è </s>
<s> Dov'è il il </s>
<s> un un </s>
Epoch 2, Update 100, Cost = 9,4009
Epoch 4, Update 200, Cost = 7,7846
Epoch 6, Update 300, Cost = 3,8434
Epoch 9, Update 400, Cost = 2,8487
Epoch 11, Update 500, Cost = 2,8454
Translations:
<s> Il cane è in al scatola. </s>
<s> Il gatto è sul scale. </s>
<s> Il gatto è sul davanzale. </s>
<s> Il gatto guarda il gatto. </s>
<s> Dov'è il cane? </s>
<s> Chi ha gatto? gatto? gatto? </s>
Epoch 2, Update 100, Cost = 10,6296
Epoch 4, Update 200, Cost = 7,9440
Epoch 6, Update 300, Cost = 13,0527
Epoch 9, Update 400, Cost = 2,0765
Epoch 11, Update 500, Cost = 4,9332
Epoch 13, Update 600, Cost = 4,1062
Epoch 15, Update 700, Cost = 7,0520
Epoch 18, Update 800, Cost = 3,9343
Translations:
<s> Il cane è in giardino. </s>
<s> Il gatto è sul tavolo. </s>
<s> Il gatto è sul letto. </s>
<s> Il gatto è in il cane. </s>
<s> Il cane gatto? </s>
<s> Canti un cane. </s>
Epoch 2, Update 100, Cost = 7,8330
Epoch 4, Update 200, Cost = 2,8366
Epoch 6, Update 300, Cost = 8,0025
Epoch 9, Update 400, Cost = 1,4480
Epoch 11, Update 500, Cost = 1,6037
Epoch 13, Update 600, Cost = 0,8747
Epoch 15, Update 700, Cost = 5,0987
Epoch 18, Update 800, Cost = 3,4031
Epoch 20, Update 900, Cost = 2,6425
Epoch 22, Update 1000, Cost = 2,2430
Epoch 24, Update 1100, Cost = 2,0058
Epoch 27, Update 1200, Cost = 1,9215
Epoch 29, Update 1300, Cost = 1,6382
Epoch 31, Update 1400, Cost = 1,6094
Epoch 34, Update 1500, Cost = 1,8286
Epoch 36, Update 1600, Cost = 1,7070
Epoch 38, Update 1700, Cost = 1,4897
Epoch 40, Update 1800, Cost = 1,4810
Epoch 43, Update 1900, Cost = 1,6980
Epoch 45, Update 2000, Cost = 1,5616
Epoch 47, Update 2100, Cost = 1,4900
Epoch 49, Update 2200, Cost = 1,4521
Epoch 52, Update 2300, Cost = 1,6467
Epoch 54, Update 2400, Cost = 1,4395
Translations:
<s> Il cane è sul pavimento. </s>
<s> Il gatto è sul davanzale. </s>
<s> Il gatto è sul davanzale. </s>
<s> Il gatto guarda l'uccello. </s>
<s> Dov'è il cane? </s>
<s> Lei ama il cane. </s>
Epoch 2, Update 100, Cost = 10,0769
Epoch 4, Update 200, Cost = 7,0053
Epoch 6, Update 300, Cost = 4,2084
Epoch 9, Update 400, Cost = 1,0056
Epoch 11, Update 500, Cost = 4,9758
Epoch 13, Update 600, Cost = 2,5352
Epoch 15, Update 700, Cost = 1,6249
Epoch 18, Update 800, Cost = 0,9074
Epoch 20, Update 900, Cost = 1,1990
Epoch 22, Update 1000, Cost = 0,8842
Epoch 24, Update 1100, Cost = 0,7719
Epoch 27, Update 1200, Cost = 0,4936
Epoch 29, Update 1300, Cost = 0,7436
Epoch 31, Update 1400, Cost = 0,6759
Epoch 34, Update 1500, Cost = 0,3836
Epoch 36, Update 1600, Cost = 0,6775
Epoch 38, Update 1700, Cost = 0,6414
Epoch 40, Update 1800, Cost = 0,6195
Epoch 43, Update 1900, Cost = 0,3638
Epoch 45, Update 2000, Cost = 0,6369
Epoch 47, Update 2100, Cost = 0,6343
Epoch 49, Update 2200, Cost = 0,6070
Epoch 52, Update 2300, Cost = 0,4008
Epoch 54, Update 2400, Cost = 0,6545
Epoch 56, Update 2500, Cost = 0,6334
Epoch 59, Update 2600, Cost = 0,3637
Epoch 61, Update 2700, Cost = 0,6535
Epoch 63, Update 2800, Cost = 0,6252
Epoch 65, Update 2900, Cost = 0,6109
Translations:
<s> Il cane è sul pavimento. </s>
<s> Il gatto è sul </s>
<s> Il mio cane è sul letto. </s>
<s> Il gatto guarda l'uccello. </s>
<s> Dov'è il cane? </s>
<s> Lei guarda il cane. </s>
Epoch 2, Update 100, Cost = 9,9098
Epoch 4, Update 200, Cost = 6,5582
Epoch 6, Update 300, Cost = 4,7191
Epoch 9, Update 400, Cost = 0,8235
Epoch 11, Update 500, Cost = 5,9987
Epoch 13, Update 600, Cost = 4,0630
Epoch 15, Update 700, Cost = 3,1621
Epoch 18, Update 800, Cost = 2,2580
Epoch 20, Update 900, Cost = 1,9802
Epoch 22, Update 1000, Cost = 1,9156
Epoch 24, Update 1100, Cost = 1,7630
Epoch 27, Update 1200, Cost = 1,4423
Epoch 29, Update 1300, Cost = 1,3841
Epoch 31, Update 1400, Cost = 1,5231
Epoch 34, Update 1500, Cost = 1,4602
Epoch 36, Update 1600, Cost = 1,3572
Epoch 38, Update 1700, Cost = 1,3241
Epoch 40, Update 1800, Cost = 1,5127
Epoch 43, Update 1900, Cost = 1,3033
Epoch 45, Update 2000, Cost = 1,3921
Epoch 47, Update 2100, Cost = 1,4603
Epoch 49, Update 2200, Cost = 1,4602
Epoch 52, Update 2300, Cost = 1,2536
Epoch 54, Update 2400, Cost = 1,2796
Epoch 56, Update 2500, Cost = 1,4435
Epoch 59, Update 2600, Cost = 1,4154
Epoch 61, Update 2700, Cost = 1,3160
Epoch 63, Update 2800, Cost = 1,3023
Epoch 65, Update 2900, Cost = 1,4959
Epoch 68, Update 3000, Cost = 1,2933
Epoch 70, Update 3100, Cost = 1,3826
Epoch 72, Update 3200, Cost = 1,4552
Epoch 74, Update 3300, Cost = 1,4566
Epoch 77, Update 3400, Cost = 1,2513
Epoch 79, Update 3500, Cost = 1,2776
Epoch 81, Update 3600, Cost = 1,4425
Epoch 84, Update 3700, Cost = 1,4148
Epoch 86, Update 3800, Cost = 1,3155
Translations:
<s> Il cane è sul pavimento. </s>
<s> Il gatto è sul </s>
<s> Il gatto dorme sul davanzale. </s>
<s> Il gatto guarda l'uccello. </s>
<s> Dov'è il cane? </s>
<s> Chi un cane. </s>
Epoch 2, Update 100, Cost = 11,2652
Epoch 4, Update 200, Cost = 13,9018
Epoch 6, Update 300, Cost = 8,4831
Epoch 9, Update 400, Cost = 3,5347
Epoch 11, Update 500, Cost = 3,1383
Epoch 13, Update 600, Cost = 2,1138
Epoch 15, Update 700, Cost = 4,3718
Epoch 18, Update 800, Cost = 2,6854
Epoch 20, Update 900, Cost = 2,8711
Epoch 22, Update 1000, Cost = 2,4394
Epoch 24, Update 1100, Cost = 2,1951
Epoch 27, Update 1200, Cost = 1,4755
Epoch 29, Update 1300, Cost = 2,1048
Epoch 31, Update 1400, Cost = 2,0237
Epoch 34, Update 1500, Cost = 1,6544
Epoch 36, Update 1600, Cost = 1,8700
Epoch 38, Update 1700, Cost = 1,8521
Epoch 40, Update 1800, Cost = 1,8595
Epoch 43, Update 1900, Cost = 1,3735
Epoch 45, Update 2000, Cost = 1,8352
Epoch 47, Update 2100, Cost = 1,7248
Epoch 49, Update 2200, Cost = 1,7633
Epoch 52, Update 2300, Cost = 1,2357
Epoch 54, Update 2400, Cost = 1,8734
Epoch 56, Update 2500, Cost = 1,8542
Epoch 59, Update 2600, Cost = 1,5783
Epoch 61, Update 2700, Cost = 1,8079
Epoch 63, Update 2800, Cost = 1,8052
Epoch 65, Update 2900, Cost = 1,8254
Epoch 68, Update 3000, Cost = 1,3595
Epoch 70, Update 3100, Cost = 1,8220
Epoch 72, Update 3200, Cost = 1,7148
Epoch 74, Update 3300, Cost = 1,7558
Epoch 77, Update 3400, Cost = 1,2328
Epoch 79, Update 3500, Cost = 1,8703
Epoch 81, Update 3600, Cost = 1,8520
Epoch 84, Update 3700, Cost = 1,5768
Epoch 86, Update 3800, Cost = 1,8072
Epoch 88, Update 3900, Cost = 1,8047
Epoch 90, Update 4000, Cost = 1,8251
Epoch 93, Update 4100, Cost = 1,3593
Epoch 95, Update 4200, Cost = 1,8220
Epoch 97, Update 4300, Cost = 1,7147
Epoch 99, Update 4400, Cost = 1,7558
Epoch 102, Update 4500, Cost = 1,2327
Epoch 104, Update 4600, Cost = 1,8703
Epoch 106, Update 4700, Cost = 1,8520
Epoch 109, Update 4800, Cost = 1,5768
Translations:
<s> Il cane è in giardino. </s>
<s> Il gatto è sul tavolo. </s>
<s> Il gatto è sul letto. </s>
<s> Il gatto guarda l'uccello. </s>
<s> Il dorme il gatto? </s>
<s> Vuoi un cane. </s>
Epoch 2, Update 100, Cost = 12,2117
Epoch 4, Update 200, Cost = 12,5486
Epoch 6, Update 300, Cost = 6,4969
Epoch 9, Update 400, Cost = 1,4105
Epoch 11, Update 500, Cost = 1,0448
Epoch 13, Update 600, Cost = 5,6472
Epoch 15, Update 700, Cost = 3,2336
Epoch 18, Update 800, Cost = 2,2628
Epoch 20, Update 900, Cost = 2,5014
Epoch 22, Update 1000, Cost = 1,6854
Epoch 24, Update 1100, Cost = 1,4931
Epoch 27, Update 1200, Cost = 1,1816
Epoch 29, Update 1300, Cost = 1,4070
Epoch 31, Update 1400, Cost = 1,3082
Epoch 34, Update 1500, Cost = 1,2662
Epoch 36, Update 1600, Cost = 1,5060
Epoch 38, Update 1700, Cost = 1,2493
Epoch 40, Update 1800, Cost = 1,2070
Epoch 43, Update 1900, Cost = 1,0138
Epoch 45, Update 2000, Cost = 1,3860
Epoch 47, Update 2100, Cost = 1,2210
Epoch 49, Update 2200, Cost = 1,1817
Epoch 52, Update 2300, Cost = 0,9860
Epoch 54, Update 2400, Cost = 1,2337
Epoch 56, Update 2500, Cost = 1,2175
Epoch 59, Update 2600, Cost = 1,2083
Epoch 61, Update 2700, Cost = 1,4514
Epoch 63, Update 2800, Cost = 1,2120
Epoch 65, Update 2900, Cost = 1,1878
Epoch 68, Update 3000, Cost = 1,0025
Epoch 70, Update 3100, Cost = 1,3737
Epoch 72, Update 3200, Cost = 1,2123
Epoch 74, Update 3300, Cost = 1,1774
Epoch 77, Update 3400, Cost = 0,9834
Epoch 79, Update 3500, Cost = 1,2313
Epoch 81, Update 3600, Cost = 1,2155
Epoch 84, Update 3700, Cost = 1,2076
Epoch 86, Update 3800, Cost = 1,4508
Epoch 88, Update 3900, Cost = 1,2116
Epoch 90, Update 4000, Cost = 1,1875
Epoch 93, Update 4100, Cost = 1,0024
Epoch 95, Update 4200, Cost = 1,3737
Epoch 97, Update 4300, Cost = 1,2123
Epoch 99, Update 4400, Cost = 1,1774
Epoch 102, Update 4500, Cost = 0,9834
Epoch 104, Update 4600, Cost = 1,2313
Epoch 106, Update 4700, Cost = 1,2155
Epoch 109, Update 4800, Cost = 1,2076
Epoch 111, Update 4900, Cost = 1,4508
Epoch 113, Update 5000, Cost = 1,2116
Epoch 115, Update 5100, Cost = 1,1875
Epoch 118, Update 5200, Cost = 1,0024
Epoch 120, Update 5300, Cost = 1,3737
Epoch 122, Update 5400, Cost = 1,2123
Epoch 124, Update 5500, Cost = 1,1774
Epoch 127, Update 5600, Cost = 0,9834
Epoch 129, Update 5700, Cost = 1,2313
Epoch 131, Update 5800, Cost = 1,2155
Epoch 134, Update 5900, Cost = 1,2076
Epoch 136, Update 6000, Cost = 1,4508
Epoch 138, Update 6100, Cost = 1,2116
Epoch 140, Update 6200, Cost = 1,1875
Epoch 143, Update 6300, Cost = 1,0024
Epoch 145, Update 6400, Cost = 1,3737
Epoch 147, Update 6500, Cost = 1,2123
Epoch 149, Update 6600, Cost = 1,1774
Epoch 152, Update 6700, Cost = 0,9834
Epoch 154, Update 6800, Cost = 1,2313
Epoch 156, Update 6900, Cost = 1,2155
Epoch 159, Update 7000, Cost = 1,2076
Epoch 161, Update 7100, Cost = 1,4508
Epoch 163, Update 7200, Cost = 1,2116
Epoch 165, Update 7300, Cost = 1,1875
Epoch 168, Update 7400, Cost = 1,0024
Epoch 170, Update 7500, Cost = 1,3737
Epoch 172, Update 7600, Cost = 1,2123
Epoch 174, Update 7700, Cost = 1,1774
Epoch 177, Update 7800, Cost = 0,9834
Epoch 179, Update 7900, Cost = 1,2313
Epoch 181, Update 8000, Cost = 1,2155
Epoch 184, Update 8100, Cost = 1,2076
Epoch 186, Update 8200, Cost = 1,4508
Epoch 188, Update 8300, Cost = 1,2116
Epoch 190, Update 8400, Cost = 1,1875
Epoch 193, Update 8500, Cost = 1,0024
Epoch 195, Update 8600, Cost = 1,3737
Epoch 197, Update 8700, Cost = 1,2123
Epoch 199, Update 8800, Cost = 1,1774
Translations:
<s> Il cane è in giardino. </s>
<s> Il gatto è sul tavolo. </s>
<s> Il gatto è sul letto. </s>
<s> Il gatto guarda l'uccello. </s>
<s> Dov'è il cane? </s>
<s> Chi un un </s>
             */

            /*
             Epoch 2, Update 100, Cost = 9,0861
Translations:
<s> Il cane è il </s>
<s> Il gatto è sul sul </s>
<s> Il gatto è sul sul </s>
<s> Il cane il il il il </s>
<s> Lui il il </s>
<s> Ha un gatto? </s>
Epoch 2, Update 100, Cost = 8,7550
Epoch 4, Update 200, Cost = 5,1205
Epoch 6, Update 300, Cost = 13,6710
Epoch 9, Update 400, Cost = 3,5471
Epoch 11, Update 500, Cost = 3,1219
Translations:
<s> Il cane è il il gatto. </s>
<s> Il gatto gatto è a casa. </s>
<s> Il gatto è sul davanzale. </s>
<s> Il cane guarda il gatto. </s>
<s> Dov'è il cane? </s>
<s> Il il il cane? </s>
Epoch 2, Update 100, Cost = 9,0027
Epoch 4, Update 200, Cost = 12,4633
Epoch 6, Update 300, Cost = 8,2133
Epoch 9, Update 400, Cost = 2,1863
Epoch 11, Update 500, Cost = 2,4530
Epoch 13, Update 600, Cost = 1,5132
Epoch 15, Update 700, Cost = 4,1096
Epoch 18, Update 800, Cost = 3,5211
Translations:
<s> Il cane è in giardino. </s>
<s> Il gatto è sul tavolo. </s>
<s> Il gatto è sul letto. </s>
<s> Il gatto guarda l'uccello. </s>
<s> Dov'è il cane? </s>
<s> Lei un cane. </s>
Epoch 2, Update 100, Cost = 8,7785
Epoch 4, Update 200, Cost = 12,1435
Epoch 6, Update 300, Cost = 5,0452
Epoch 9, Update 400, Cost = 0,6328
Epoch 11, Update 500, Cost = 0,8218
Epoch 13, Update 600, Cost = 0,5547
Epoch 15, Update 700, Cost = 3,2468
Epoch 18, Update 800, Cost = 1,9823
Epoch 20, Update 900, Cost = 2,2827
Epoch 22, Update 1000, Cost = 1,7130
Epoch 24, Update 1100, Cost = 1,2878
Epoch 27, Update 1200, Cost = 1,2205
Epoch 29, Update 1300, Cost = 1,2427
Epoch 31, Update 1400, Cost = 1,1551
Epoch 34, Update 1500, Cost = 0,6399
Epoch 36, Update 1600, Cost = 1,3343
Epoch 38, Update 1700, Cost = 1,0905
Epoch 40, Update 1800, Cost = 1,0159
Epoch 43, Update 1900, Cost = 0,8338
Epoch 45, Update 2000, Cost = 1,1854
Epoch 47, Update 2100, Cost = 1,0407
Epoch 49, Update 2200, Cost = 0,9908
Epoch 52, Update 2300, Cost = 1,0088
Epoch 54, Update 2400, Cost = 1,0751
Translations:
<s> Il cane è in giardino. </s>
<s> Il gatto è sul tavolo. </s>
<s> Il gatto dorme sul davanzale. </s>
<s> Il gatto guarda l'uccello. </s>
<s> Dov'è il cane? </s>
<s> Chi un un gatto? </s>
Epoch 2, Update 100, Cost = 8,8902
Epoch 4, Update 200, Cost = 4,9758
Epoch 6, Update 300, Cost = 3,9500
Epoch 9, Update 400, Cost = 10,6191
Epoch 11, Update 500, Cost = 5,8212
Epoch 13, Update 600, Cost = 3,7410
Epoch 15, Update 700, Cost = 2,8851
Epoch 18, Update 800, Cost = 1,0436
Epoch 20, Update 900, Cost = 3,7303
Epoch 22, Update 1000, Cost = 2,8739
Epoch 24, Update 1100, Cost = 2,4857
Epoch 27, Update 1200, Cost = 2,3073
Epoch 29, Update 1300, Cost = 1,8466
Epoch 31, Update 1400, Cost = 1,8955
Epoch 34, Update 1500, Cost = 1,7743
Epoch 36, Update 1600, Cost = 2,0367
Epoch 38, Update 1700, Cost = 1,6897
Epoch 40, Update 1800, Cost = 1,6648
Epoch 43, Update 1900, Cost = 1,5965
Epoch 45, Update 2000, Cost = 1,7526
Epoch 47, Update 2100, Cost = 1,6868
Epoch 49, Update 2200, Cost = 1,6546
Epoch 52, Update 2300, Cost = 1,6187
Epoch 54, Update 2400, Cost = 1,5880
Epoch 56, Update 2500, Cost = 1,7060
Epoch 59, Update 2600, Cost = 1,6373
Epoch 61, Update 2700, Cost = 1,9163
Epoch 63, Update 2800, Cost = 1,6119
Epoch 65, Update 2900, Cost = 1,6286
Translations:
<s> Il cane è in giardino. </s>
<s> Il gatto è sul tavolo. </s>
<s> Il gatto è sul letto. </s>
<s> Il gatto guarda l'uccello. </s>
<s> Il cane è gatto? </s>
<s> Hai un cane. </s>
Epoch 2, Update 100, Cost = 11,7670
Epoch 4, Update 200, Cost = 7,4487
Epoch 6, Update 300, Cost = 4,0071
Epoch 9, Update 400, Cost = 0,5444
Epoch 11, Update 500, Cost = 0,9959
Epoch 13, Update 600, Cost = 4,9133
Epoch 15, Update 700, Cost = 3,1041
Epoch 18, Update 800, Cost = 2,6569
Epoch 20, Update 900, Cost = 2,3075
Epoch 22, Update 1000, Cost = 1,9847
Epoch 24, Update 1100, Cost = 1,6050
Epoch 27, Update 1200, Cost = 1,5785
Epoch 29, Update 1300, Cost = 1,4735
Epoch 31, Update 1400, Cost = 1,4495
Epoch 34, Update 1500, Cost = 1,4752
Epoch 36, Update 1600, Cost = 1,5612
Epoch 38, Update 1700, Cost = 1,3892
Epoch 40, Update 1800, Cost = 1,2911
Epoch 43, Update 1900, Cost = 1,3033
Epoch 45, Update 2000, Cost = 1,3987
Epoch 47, Update 2100, Cost = 1,3993
Epoch 49, Update 2200, Cost = 1,2883
Epoch 52, Update 2300, Cost = 1,3426
Epoch 54, Update 2400, Cost = 1,3016
Epoch 56, Update 2500, Cost = 1,3304
Epoch 59, Update 2600, Cost = 1,4134
Epoch 61, Update 2700, Cost = 1,5122
Epoch 63, Update 2800, Cost = 1,3561
Epoch 65, Update 2900, Cost = 1,2693
Epoch 68, Update 3000, Cost = 1,2904
Epoch 70, Update 3100, Cost = 1,3886
Epoch 72, Update 3200, Cost = 1,3921
Epoch 74, Update 3300, Cost = 1,2835
Epoch 77, Update 3400, Cost = 1,3399
Epoch 79, Update 3500, Cost = 1,2997
Epoch 81, Update 3600, Cost = 1,3291
Epoch 84, Update 3700, Cost = 1,4123
Epoch 86, Update 3800, Cost = 1,5114
Translations:
<s> Il cane è sul pavimento. </s>
<s> Il gatto è sul </s>
<s> Il gatto dorme sul davanzale. </s>
<s> Il gatto guarda l'uccello. </s>
<s> Dov'è il cane? </s>
<s> Lei ama il cane. </s>
Epoch 2, Update 100, Cost = 7,5965
Epoch 4, Update 200, Cost = 13,3677
Epoch 6, Update 300, Cost = 6,6512
Epoch 9, Update 400, Cost = 0,9584
Epoch 11, Update 500, Cost = 1,0381
Epoch 13, Update 600, Cost = 0,6780
Epoch 15, Update 700, Cost = 3,3391
Epoch 18, Update 800, Cost = 2,4685
Epoch 20, Update 900, Cost = 2,3399
Epoch 22, Update 1000, Cost = 1,8746
Epoch 24, Update 1100, Cost = 1,6436
Epoch 27, Update 1200, Cost = 1,8688
Epoch 29, Update 1300, Cost = 1,5411
Epoch 31, Update 1400, Cost = 1,4363
Epoch 34, Update 1500, Cost = 1,6381
Epoch 36, Update 1600, Cost = 1,6926
Epoch 38, Update 1700, Cost = 1,3333
Epoch 40, Update 1800, Cost = 1,2612
Epoch 43, Update 1900, Cost = 1,4656
Epoch 45, Update 2000, Cost = 1,4661
Epoch 47, Update 2100, Cost = 1,3162
Epoch 49, Update 2200, Cost = 1,2209
Epoch 52, Update 2300, Cost = 1,5401
Epoch 54, Update 2400, Cost = 1,3203
Epoch 56, Update 2500, Cost = 1,2860
Epoch 59, Update 2600, Cost = 1,5559
Epoch 61, Update 2700, Cost = 1,6304
Epoch 63, Update 2800, Cost = 1,2985
Epoch 65, Update 2900, Cost = 1,2374
Epoch 68, Update 3000, Cost = 1,4391
Epoch 70, Update 3100, Cost = 1,4540
Epoch 72, Update 3200, Cost = 1,3088
Epoch 74, Update 3300, Cost = 1,2157
Epoch 77, Update 3400, Cost = 1,5344
Epoch 79, Update 3500, Cost = 1,3180
Epoch 81, Update 3600, Cost = 1,2845
Epoch 84, Update 3700, Cost = 1,5543
Epoch 86, Update 3800, Cost = 1,6293
Epoch 88, Update 3900, Cost = 1,2978
Epoch 90, Update 4000, Cost = 1,2372
Epoch 93, Update 4100, Cost = 1,4390
Epoch 95, Update 4200, Cost = 1,4539
Epoch 97, Update 4300, Cost = 1,3087
Epoch 99, Update 4400, Cost = 1,2157
Epoch 102, Update 4500, Cost = 1,5344
Epoch 104, Update 4600, Cost = 1,3180
Epoch 106, Update 4700, Cost = 1,2845
Epoch 109, Update 4800, Cost = 1,5543
Translations:
<s> Il cane è in nella </s>
<s> Il gatto dorme sul </s>
<s> Il gatto è sul davanzale. </s>
<s> Il gatto guarda l'uccello. </s>
<s> Dov'è il cane? </s>
<s> Chi ha un </s>
Epoch 2, Update 100, Cost = 10,3772
Epoch 4, Update 200, Cost = 10,4091
Epoch 6, Update 300, Cost = 7,0826
Epoch 9, Update 400, Cost = 1,6994
Epoch 11, Update 500, Cost = 2,8826
Epoch 13, Update 600, Cost = 6,3276
Epoch 15, Update 700, Cost = 4,7165
Epoch 18, Update 800, Cost = 3,8249
Epoch 20, Update 900, Cost = 3,5604
Epoch 22, Update 1000, Cost = 3,6548
Epoch 24, Update 1100, Cost = 3,1193
Epoch 27, Update 1200, Cost = 3,3537
Epoch 29, Update 1300, Cost = 2,8701
Epoch 31, Update 1400, Cost = 2,8533
Epoch 34, Update 1500, Cost = 2,8862
Epoch 36, Update 1600, Cost = 3,0640
Epoch 38, Update 1700, Cost = 2,7221
Epoch 40, Update 1800, Cost = 2,5973
Epoch 43, Update 1900, Cost = 2,8053
Epoch 45, Update 2000, Cost = 2,7563
Epoch 47, Update 2100, Cost = 2,8132
Epoch 49, Update 2200, Cost = 2,6060
Epoch 52, Update 2300, Cost = 2,9229
Epoch 54, Update 2400, Cost = 2,5388
Epoch 56, Update 2500, Cost = 2,7183
Epoch 59, Update 2600, Cost = 2,8165
Epoch 61, Update 2700, Cost = 2,9697
Epoch 63, Update 2800, Cost = 2,6599
Epoch 65, Update 2900, Cost = 2,5698
Epoch 68, Update 3000, Cost = 2,7865
Epoch 70, Update 3100, Cost = 2,7372
Epoch 72, Update 3200, Cost = 2,7991
Epoch 74, Update 3300, Cost = 2,5999
Epoch 77, Update 3400, Cost = 2,9175
Epoch 79, Update 3500, Cost = 2,5348
Epoch 81, Update 3600, Cost = 2,7154
Epoch 84, Update 3700, Cost = 2,8156
Epoch 86, Update 3800, Cost = 2,9686
Epoch 88, Update 3900, Cost = 2,6593
Epoch 90, Update 4000, Cost = 2,5694
Epoch 93, Update 4100, Cost = 2,7864
Epoch 95, Update 4200, Cost = 2,7372
Epoch 97, Update 4300, Cost = 2,7990
Epoch 99, Update 4400, Cost = 2,5999
Epoch 102, Update 4500, Cost = 2,9174
Epoch 104, Update 4600, Cost = 2,5348
Epoch 106, Update 4700, Cost = 2,7154
Epoch 109, Update 4800, Cost = 2,8156
Epoch 111, Update 4900, Cost = 2,9686
Epoch 113, Update 5000, Cost = 2,6593
Epoch 115, Update 5100, Cost = 2,5694
Epoch 118, Update 5200, Cost = 2,7864
Epoch 120, Update 5300, Cost = 2,7372
Epoch 122, Update 5400, Cost = 2,7990
Epoch 124, Update 5500, Cost = 2,5999
Epoch 127, Update 5600, Cost = 2,9174
Epoch 129, Update 5700, Cost = 2,5348
Epoch 131, Update 5800, Cost = 2,7154
Epoch 134, Update 5900, Cost = 2,8156
Epoch 136, Update 6000, Cost = 2,9686
Epoch 138, Update 6100, Cost = 2,6593
Epoch 140, Update 6200, Cost = 2,5694
Epoch 143, Update 6300, Cost = 2,7864
Epoch 145, Update 6400, Cost = 2,7372
Epoch 147, Update 6500, Cost = 2,7990
Epoch 149, Update 6600, Cost = 2,5999
Epoch 152, Update 6700, Cost = 2,9174
Epoch 154, Update 6800, Cost = 2,5348
Epoch 156, Update 6900, Cost = 2,7154
Epoch 159, Update 7000, Cost = 2,8156
Epoch 161, Update 7100, Cost = 2,9686
Epoch 163, Update 7200, Cost = 2,6593
Epoch 165, Update 7300, Cost = 2,5694
Epoch 168, Update 7400, Cost = 2,7864
Epoch 170, Update 7500, Cost = 2,7372
Epoch 172, Update 7600, Cost = 2,7990
Epoch 174, Update 7700, Cost = 2,5999
Epoch 177, Update 7800, Cost = 2,9174
Epoch 179, Update 7900, Cost = 2,5348
Epoch 181, Update 8000, Cost = 2,7154
Epoch 184, Update 8100, Cost = 2,8156
Epoch 186, Update 8200, Cost = 2,9686
Epoch 188, Update 8300, Cost = 2,6593
Epoch 190, Update 8400, Cost = 2,5694
Epoch 193, Update 8500, Cost = 2,7864
Epoch 195, Update 8600, Cost = 2,7372
Epoch 197, Update 8700, Cost = 2,7990
Epoch 199, Update 8800, Cost = 2,5999
Translations:
<s> Il cane è sul il </s>
<s> Il gatto è sul tavolo. </s>
<s> Il gatto è sul </s>
<s> Il gatto guarda l'uccello. </s>
<s> Dov'è il cane? </s>
<s> Lei un gatto? </s>
             */

            Console.ReadLine();
        }
    }
}
