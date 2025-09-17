using Seq2SeqSharp;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Enums;
using Seq2SeqSharp.LearningRate;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;

// ChatGPT, modified
// Application uses 1 GigaByte of RAM. 
// 2 layers
namespace ConsoleDutchItalianSeq2Seq15Jul2025
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
                    EncoderLayerDepth = 2,
                    DecoderLayerDepth = 2,
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
Epoch 2, Update 100, Cost = 13,8536
Translations:
<s> Il cane è è </s>
<s> Il gatto è il </s>
<s> Il gatto è è </s>
<s> Il gatto il il </s>
<s> Il cane è </s>
<s> il il </s>
Epoch 2, Update 100, Cost = 11,9190
Epoch 4, Update 200, Cost = 12,4703
Epoch 6, Update 300, Cost = 10,7812
Epoch 9, Update 400, Cost = 4,7334
Epoch 11, Update 500, Cost = 5,7108
Translations:
<s> Il cane è in giardino. </s>
<s> Il gatto è sul </s>
<s> Il gatto è sul sedia. </s>
<s> Il gatto guarda l'uccello. </s>
<s> Dov'è il cane? </s>
<s> Chi ha un gatto? </s>
Epoch 2, Update 100, Cost = 11,6828
Epoch 4, Update 200, Cost = 11,7509
Epoch 6, Update 300, Cost = 9,8011
Epoch 9, Update 400, Cost = 4,3306
Epoch 11, Update 500, Cost = 3,7717
Epoch 13, Update 600, Cost = 3,0485
Epoch 15, Update 700, Cost = 2,5446
Epoch 18, Update 800, Cost = 1,6406
Translations:
<s> Il cane è in giardino. </s>
<s> Il gatto è sul </s>
<s> Lei gatto sul davanzale. </s>
<s> Il gatto guarda il cane. </s>
<s> Il due tuo </s>
<s> Hai un cane. cane. </s>
Epoch 2, Update 100, Cost = 11,0933
Epoch 4, Update 200, Cost = 10,9971
Epoch 6, Update 300, Cost = 9,2576
Epoch 9, Update 400, Cost = 5,1757
Epoch 11, Update 500, Cost = 5,1565
Epoch 13, Update 600, Cost = 3,7047
Epoch 15, Update 700, Cost = 3,1316
Epoch 18, Update 800, Cost = 2,3570
Epoch 20, Update 900, Cost = 2,0884
Epoch 22, Update 1000, Cost = 2,0469
Epoch 24, Update 1100, Cost = 1,9480
Epoch 27, Update 1200, Cost = 1,8195
Epoch 29, Update 1300, Cost = 1,4269
Epoch 31, Update 1400, Cost = 1,6956
Epoch 34, Update 1500, Cost = 1,5318
Epoch 36, Update 1600, Cost = 1,5041
Epoch 38, Update 1700, Cost = 1,4528
Epoch 40, Update 1800, Cost = 1,6275
Epoch 43, Update 1900, Cost = 1,5194
Epoch 45, Update 2000, Cost = 1,5016
Epoch 47, Update 2100, Cost = 1,5858
Epoch 49, Update 2200, Cost = 1,6351
Epoch 52, Update 2300, Cost = 1,6319
Epoch 54, Update 2400, Cost = 1,3268
Translations:
<s> Il cane è in giardino. </s>
<s> Il gatto è sul </s>
<s> Lei dorme sul sedia. </s>
<s> Lei gatto il cane. </s>
<s> Abbiamo dorme gatto? </s>
<s> Hai un cane. </s>
Epoch 2, Update 100, Cost = 12,3081
Epoch 4, Update 200, Cost = 10,8724
Epoch 6, Update 300, Cost = 9,1200
Epoch 9, Update 400, Cost = 5,2104
Epoch 11, Update 500, Cost = 5,5890
Epoch 13, Update 600, Cost = 4,8915
Epoch 15, Update 700, Cost = 3,2344
Epoch 18, Update 800, Cost = 2,6118
Epoch 20, Update 900, Cost = 2,1515
Epoch 22, Update 1000, Cost = 1,9228
Epoch 24, Update 1100, Cost = 1,7148
Epoch 27, Update 1200, Cost = 1,5161
Epoch 29, Update 1300, Cost = 1,4118
Epoch 31, Update 1400, Cost = 1,3895
Epoch 34, Update 1500, Cost = 1,4042
Epoch 36, Update 1600, Cost = 1,3762
Epoch 38, Update 1700, Cost = 1,3818
Epoch 40, Update 1800, Cost = 1,2959
Epoch 43, Update 1900, Cost = 1,3678
Epoch 45, Update 2000, Cost = 1,3165
Epoch 47, Update 2100, Cost = 1,3439
Epoch 49, Update 2200, Cost = 1,3477
Epoch 52, Update 2300, Cost = 1,2555
Epoch 54, Update 2400, Cost = 1,2584
Epoch 56, Update 2500, Cost = 1,2779
Epoch 59, Update 2600, Cost = 1,3307
Epoch 61, Update 2700, Cost = 1,3239
Epoch 63, Update 2800, Cost = 1,3477
Epoch 65, Update 2900, Cost = 1,2736
Translations:
<s> Il cane è in giardino. </s>
<s> Il gatto è sul tavolo. </s>
<s> Il gatto è sul sul </s>
<s> Il gatto guarda l'uccello. </s>
<s> Dov'è il cane? </s>
<s> Lei guarda il film. </s>
Epoch 2, Update 100, Cost = 11,4315
Epoch 4, Update 200, Cost = 11,2667
Epoch 6, Update 300, Cost = 9,3909
Epoch 9, Update 400, Cost = 3,8410
Epoch 11, Update 500, Cost = 4,4753
Epoch 13, Update 600, Cost = 3,2565
Epoch 15, Update 700, Cost = 2,7396
Epoch 18, Update 800, Cost = 1,2758
Epoch 20, Update 900, Cost = 1,4973
Epoch 22, Update 1000, Cost = 1,2052
Epoch 24, Update 1100, Cost = 1,3474
Epoch 27, Update 1200, Cost = 0,8074
Epoch 29, Update 1300, Cost = 1,0020
Epoch 31, Update 1400, Cost = 1,0972
Epoch 34, Update 1500, Cost = 0,7767
Epoch 36, Update 1600, Cost = 0,9100
Epoch 38, Update 1700, Cost = 0,9000
Epoch 40, Update 1800, Cost = 1,0362
Epoch 43, Update 1900, Cost = 0,6321
Epoch 45, Update 2000, Cost = 0,9036
Epoch 47, Update 2100, Cost = 0,8206
Epoch 49, Update 2200, Cost = 1,0163
Epoch 52, Update 2300, Cost = 0,6729
Epoch 54, Update 2400, Cost = 0,8850
Epoch 56, Update 2500, Cost = 0,9960
Epoch 59, Update 2600, Cost = 0,7340
Epoch 61, Update 2700, Cost = 0,8686
Epoch 63, Update 2800, Cost = 0,8736
Epoch 65, Update 2900, Cost = 1,0141
Epoch 68, Update 3000, Cost = 0,6245
Epoch 70, Update 3100, Cost = 0,8954
Epoch 72, Update 3200, Cost = 0,8154
Epoch 74, Update 3300, Cost = 1,0119
Epoch 77, Update 3400, Cost = 0,6710
Epoch 79, Update 3500, Cost = 0,8833
Epoch 81, Update 3600, Cost = 0,9947
Epoch 84, Update 3700, Cost = 0,7335
Epoch 86, Update 3800, Cost = 0,8682
Translations:
<s> Il cane è in giardino. </s>
<s> Il gatto è sul tavolo. </s>
<s> Il suo gatto sul davanzale. </s>
<s> Il gatto guarda l'uccello. </s>
<s> Dov'è il il gatto? </s>
<s> Hai un cane. </s>
Epoch 2, Update 100, Cost = 13,3114
Epoch 4, Update 200, Cost = 13,8680
Epoch 6, Update 300, Cost = 11,5751
Epoch 9, Update 400, Cost = 5,4263
Epoch 11, Update 500, Cost = 4,9703
Epoch 13, Update 600, Cost = 3,3572
Epoch 15, Update 700, Cost = 3,0021
Epoch 18, Update 800, Cost = 1,1079
Epoch 20, Update 900, Cost = 1,2413
Epoch 22, Update 1000, Cost = 1,0144
Epoch 24, Update 1100, Cost = 1,4091
Epoch 27, Update 1200, Cost = 0,6368
Epoch 29, Update 1300, Cost = 0,7154
Epoch 31, Update 1400, Cost = 1,0445
Epoch 34, Update 1500, Cost = 0,4326
Epoch 36, Update 1600, Cost = 0,6784
Epoch 38, Update 1700, Cost = 0,6816
Epoch 40, Update 1800, Cost = 0,9949
Epoch 43, Update 1900, Cost = 0,3844
Epoch 45, Update 2000, Cost = 0,6897
Epoch 47, Update 2100, Cost = 0,6549
Epoch 49, Update 2200, Cost = 1,0711
Epoch 52, Update 2300, Cost = 0,5197
Epoch 54, Update 2400, Cost = 0,6268
Epoch 56, Update 2500, Cost = 0,9582
Epoch 59, Update 2600, Cost = 0,3999
Epoch 61, Update 2700, Cost = 0,6500
Epoch 63, Update 2800, Cost = 0,6616
Epoch 65, Update 2900, Cost = 0,9753
Epoch 68, Update 3000, Cost = 0,3785
Epoch 70, Update 3100, Cost = 0,6835
Epoch 72, Update 3200, Cost = 0,6507
Epoch 74, Update 3300, Cost = 1,0668
Epoch 77, Update 3400, Cost = 0,5182
Epoch 79, Update 3500, Cost = 0,6256
Epoch 81, Update 3600, Cost = 0,9571
Epoch 84, Update 3700, Cost = 0,3995
Epoch 86, Update 3800, Cost = 0,6498
Epoch 88, Update 3900, Cost = 0,6615
Epoch 90, Update 4000, Cost = 0,9752
Epoch 93, Update 4100, Cost = 0,3784
Epoch 95, Update 4200, Cost = 0,6835
Epoch 97, Update 4300, Cost = 0,6507
Epoch 99, Update 4400, Cost = 1,0668
Epoch 102, Update 4500, Cost = 0,5182
Epoch 104, Update 4600, Cost = 0,6256
Epoch 106, Update 4700, Cost = 0,9571
Epoch 109, Update 4800, Cost = 0,3995
Translations:
<s> Il cane è in giardino. </s>
<s> Il gatto è sul tavolo. </s>
<s> Il gatto è sul letto. </s>
<s> Il gatto è accanto cane. </s>
<s> Ha mangia gatto? </s>
<s> Hai un cane. </s>
Epoch 2, Update 100, Cost = 12,8773
Epoch 4, Update 200, Cost = 10,8467
Epoch 6, Update 300, Cost = 9,5217
Epoch 9, Update 400, Cost = 4,5544
Epoch 11, Update 500, Cost = 3,6113
Epoch 13, Update 600, Cost = 2,4573
Epoch 15, Update 700, Cost = 1,8797
Epoch 18, Update 800, Cost = 0,6836
Epoch 20, Update 900, Cost = 0,7923
Epoch 22, Update 1000, Cost = 0,7770
Epoch 24, Update 1100, Cost = 0,6908
Epoch 27, Update 1200, Cost = 0,3527
Epoch 29, Update 1300, Cost = 0,5196
Epoch 31, Update 1400, Cost = 0,5544
Epoch 34, Update 1500, Cost = 0,2471
Epoch 36, Update 1600, Cost = 0,4312
Epoch 38, Update 1700, Cost = 0,4710
Epoch 40, Update 1800, Cost = 0,4917
Epoch 43, Update 1900, Cost = 0,2242
Epoch 45, Update 2000, Cost = 0,4171
Epoch 47, Update 2100, Cost = 0,4812
Epoch 49, Update 2200, Cost = 0,4921
Epoch 52, Update 2300, Cost = 0,2706
Epoch 54, Update 2400, Cost = 0,4473
Epoch 56, Update 2500, Cost = 0,4938
Epoch 59, Update 2600, Cost = 0,2233
Epoch 61, Update 2700, Cost = 0,4101
Epoch 63, Update 2800, Cost = 0,4552
Epoch 65, Update 2900, Cost = 0,4794
Epoch 68, Update 3000, Cost = 0,2197
Epoch 70, Update 3100, Cost = 0,4125
Epoch 72, Update 3200, Cost = 0,4773
Epoch 74, Update 3300, Cost = 0,4894
Epoch 77, Update 3400, Cost = 0,2695
Epoch 79, Update 3500, Cost = 0,4464
Epoch 81, Update 3600, Cost = 0,4930
Epoch 84, Update 3700, Cost = 0,2231
Epoch 86, Update 3800, Cost = 0,4098
Epoch 88, Update 3900, Cost = 0,4551
Epoch 90, Update 4000, Cost = 0,4793
Epoch 93, Update 4100, Cost = 0,2197
Epoch 95, Update 4200, Cost = 0,4125
Epoch 97, Update 4300, Cost = 0,4772
Epoch 99, Update 4400, Cost = 0,4894
Epoch 102, Update 4500, Cost = 0,2695
Epoch 104, Update 4600, Cost = 0,4464
Epoch 106, Update 4700, Cost = 0,4930
Epoch 109, Update 4800, Cost = 0,2231
Epoch 111, Update 4900, Cost = 0,4098
Epoch 113, Update 5000, Cost = 0,4551
Epoch 115, Update 5100, Cost = 0,4793
Epoch 118, Update 5200, Cost = 0,2197
Epoch 120, Update 5300, Cost = 0,4125
Epoch 122, Update 5400, Cost = 0,4772
Epoch 124, Update 5500, Cost = 0,4894
Epoch 127, Update 5600, Cost = 0,2695
Epoch 129, Update 5700, Cost = 0,4464
Epoch 131, Update 5800, Cost = 0,4930
Epoch 134, Update 5900, Cost = 0,2231
Epoch 136, Update 6000, Cost = 0,4098
Epoch 138, Update 6100, Cost = 0,4551
Epoch 140, Update 6200, Cost = 0,4793
Epoch 143, Update 6300, Cost = 0,2197
Epoch 145, Update 6400, Cost = 0,4125
Epoch 147, Update 6500, Cost = 0,4772
Epoch 149, Update 6600, Cost = 0,4894
Epoch 152, Update 6700, Cost = 0,2695
Epoch 154, Update 6800, Cost = 0,4464
Epoch 156, Update 6900, Cost = 0,4930
Epoch 159, Update 7000, Cost = 0,2231
Epoch 161, Update 7100, Cost = 0,4098
Epoch 163, Update 7200, Cost = 0,4551
Epoch 165, Update 7300, Cost = 0,4793
Epoch 168, Update 7400, Cost = 0,2197
Epoch 170, Update 7500, Cost = 0,4125
Epoch 172, Update 7600, Cost = 0,4772
Epoch 174, Update 7700, Cost = 0,4894
Epoch 177, Update 7800, Cost = 0,2695
Epoch 179, Update 7900, Cost = 0,4464
Epoch 181, Update 8000, Cost = 0,4930
Epoch 184, Update 8100, Cost = 0,2231
Epoch 186, Update 8200, Cost = 0,4098
Epoch 188, Update 8300, Cost = 0,4551
Epoch 190, Update 8400, Cost = 0,4793
Epoch 193, Update 8500, Cost = 0,2197
Epoch 195, Update 8600, Cost = 0,4125
Epoch 197, Update 8700, Cost = 0,4772
Epoch 199, Update 8800, Cost = 0,4894
Translations:
<s> Il cane è in giardino. </s>
<s> Il gatto è sul tavolo. </s>
<s> Il mio è sul sul </s>
<s> Il gatto guarda l'uccello. </s>
<s> Dov'è il gatto? </s>
<s> Lei ama il film. </s>
            */

            /*
 Epoch 2, Update 100, Cost = 10,8827
Translations:
<s> Il cane cane il </s>
<s> Il gatto è il </s>
<s> Il gatto gatto è </s>
<s> Il gatto gatto il </s>
<s> Lei il </s>
<s> </s>
Epoch 2, Update 100, Cost = 12,0593
Epoch 4, Update 200, Cost = 12,7434
Epoch 6, Update 300, Cost = 10,1584
Epoch 9, Update 400, Cost = 8,6530
Epoch 11, Update 500, Cost = 8,5096
Translations:
<s> Il cane è in in </s>
<s> Il gatto è sul al </s>
<s> Il gatto gatto gatto gatto? </s>
<s> Il gatto guarda il gatto. </s>
<s> Il loro è </s>
<s> Lei un il gatto? </s>
Epoch 2, Update 100, Cost = 11,8376
Epoch 4, Update 200, Cost = 10,4696
Epoch 6, Update 300, Cost = 9,9810
Epoch 9, Update 400, Cost = 5,3014
Epoch 11, Update 500, Cost = 6,3551
Epoch 13, Update 600, Cost = 4,7801
Epoch 15, Update 700, Cost = 4,6757
Epoch 18, Update 800, Cost = 2,7656
Translations:
<s> Il cane è in in </s>
<s> Il gatto è sul sedia. </s>
<s> Il gatto è sul sedia. </s>
<s> Il gatto è l'uccello. </s>
<s> Dov'è il cane? </s>
<s> Lei guarda il cane. </s>
Epoch 2, Update 100, Cost = 11,1827
Epoch 4, Update 200, Cost = 9,0378
Epoch 6, Update 300, Cost = 9,1935
Epoch 9, Update 400, Cost = 5,9739
Epoch 11, Update 500, Cost = 5,4722
Epoch 13, Update 600, Cost = 4,3477
Epoch 15, Update 700, Cost = 3,6931
Epoch 18, Update 800, Cost = 3,2123
Epoch 20, Update 900, Cost = 2,6095
Epoch 22, Update 1000, Cost = 2,4227
Epoch 24, Update 1100, Cost = 2,2914
Epoch 27, Update 1200, Cost = 2,3785
Epoch 29, Update 1300, Cost = 1,8297
Epoch 31, Update 1400, Cost = 1,9985
Epoch 34, Update 1500, Cost = 2,1683
Epoch 36, Update 1600, Cost = 2,0381
Epoch 38, Update 1700, Cost = 1,8754
Epoch 40, Update 1800, Cost = 1,9049
Epoch 43, Update 1900, Cost = 2,2029
Epoch 45, Update 2000, Cost = 1,9317
Epoch 47, Update 2100, Cost = 1,9584
Epoch 49, Update 2200, Cost = 1,9607
Epoch 52, Update 2300, Cost = 2,1561
Epoch 54, Update 2400, Cost = 1,7031
Translations:
<s> Il cane è in giardino. </s>
<s> Il gatto è sul tavolo. </s>
<s> Giochiamo al parco. </s>
<s> Il gatto è al cane. </s>
<s> Dov'è nostro il gatto? </s>
<s> Hai un cane. </s>
Epoch 2, Update 100, Cost = 10,9501
Epoch 4, Update 200, Cost = 9,4979
Epoch 6, Update 300, Cost = 9,8105
Epoch 9, Update 400, Cost = 5,0486
Epoch 11, Update 500, Cost = 4,9270
Epoch 13, Update 600, Cost = 3,7301
Epoch 15, Update 700, Cost = 3,2670
Epoch 18, Update 800, Cost = 2,3531
Epoch 20, Update 900, Cost = 2,1301
Epoch 22, Update 1000, Cost = 2,0233
Epoch 24, Update 1100, Cost = 1,9560
Epoch 27, Update 1200, Cost = 1,7018
Epoch 29, Update 1300, Cost = 1,4859
Epoch 31, Update 1400, Cost = 1,6382
Epoch 34, Update 1500, Cost = 1,8483
Epoch 36, Update 1600, Cost = 1,5165
Epoch 38, Update 1700, Cost = 1,5117
Epoch 40, Update 1800, Cost = 1,6236
Epoch 43, Update 1900, Cost = 1,6042
Epoch 45, Update 2000, Cost = 1,5710
Epoch 47, Update 2100, Cost = 1,6033
Epoch 49, Update 2200, Cost = 1,6565
Epoch 52, Update 2300, Cost = 1,5468
Epoch 54, Update 2400, Cost = 1,3883
Epoch 56, Update 2500, Cost = 1,5545
Epoch 59, Update 2600, Cost = 1,8063
Epoch 61, Update 2700, Cost = 1,4838
Epoch 63, Update 2800, Cost = 1,4862
Epoch 65, Update 2900, Cost = 1,6058
Translations:
<s> Il cane è il il </s>
<s> Il gatto mangia pesce. </s>
<s> Il gatto dorme sul </s>
<s> Il gatto è accanto cane. cane. </s>
<s> Dov'è il cane? </s>
<s> Chi un un </s>
Epoch 2, Update 100, Cost = 10,6577
Epoch 4, Update 200, Cost = 13,1255
Epoch 6, Update 300, Cost = 10,0211
Epoch 9, Update 400, Cost = 7,4475
Epoch 11, Update 500, Cost = 6,0240
Epoch 13, Update 600, Cost = 5,9313
Epoch 15, Update 700, Cost = 4,1174
Epoch 18, Update 800, Cost = 2,9391
Epoch 20, Update 900, Cost = 2,5777
Epoch 22, Update 1000, Cost = 2,4363
Epoch 24, Update 1100, Cost = 2,2079
Epoch 27, Update 1200, Cost = 1,8903
Epoch 29, Update 1300, Cost = 1,7160
Epoch 31, Update 1400, Cost = 1,8988
Epoch 34, Update 1500, Cost = 2,0368
Epoch 36, Update 1600, Cost = 1,5741
Epoch 38, Update 1700, Cost = 1,6964
Epoch 40, Update 1800, Cost = 1,8227
Epoch 43, Update 1900, Cost = 1,7601
Epoch 45, Update 2000, Cost = 1,7106
Epoch 47, Update 2100, Cost = 1,8035
Epoch 49, Update 2200, Cost = 1,8235
Epoch 52, Update 2300, Cost = 1,6487
Epoch 54, Update 2400, Cost = 1,5676
Epoch 56, Update 2500, Cost = 1,7727
Epoch 59, Update 2600, Cost = 1,9711
Epoch 61, Update 2700, Cost = 1,5203
Epoch 63, Update 2800, Cost = 1,6620
Epoch 65, Update 2900, Cost = 1,7962
Epoch 68, Update 3000, Cost = 1,7460
Epoch 70, Update 3100, Cost = 1,6991
Epoch 72, Update 3200, Cost = 1,7942
Epoch 74, Update 3300, Cost = 1,8177
Epoch 77, Update 3400, Cost = 1,6448
Epoch 79, Update 3500, Cost = 1,5654
Epoch 81, Update 3600, Cost = 1,7708
Epoch 84, Update 3700, Cost = 1,9702
Epoch 86, Update 3800, Cost = 1,5196
Translations:
<s> Il cane è sul pavimento. </s>
<s> Il gatto è sul </s>
<s> Il gatto gatto sul davanzale. </s>
<s> Il gatto guarda l'uccello. </s>
<s> Dov'è il cane? </s>
<s> Chi ha un gatto? </s>
Epoch 2, Update 100, Cost = 11,1488
Epoch 4, Update 200, Cost = 9,6653
Epoch 6, Update 300, Cost = 10,3318
Epoch 9, Update 400, Cost = 4,8731
Epoch 11, Update 500, Cost = 4,4651
Epoch 13, Update 600, Cost = 3,9052
Epoch 15, Update 700, Cost = 3,5231
Epoch 18, Update 800, Cost = 2,3077
Epoch 20, Update 900, Cost = 2,1076
Epoch 22, Update 1000, Cost = 2,0974
Epoch 24, Update 1100, Cost = 2,0148
Epoch 27, Update 1200, Cost = 1,5461
Epoch 29, Update 1300, Cost = 1,4974
Epoch 31, Update 1400, Cost = 1,7134
Epoch 34, Update 1500, Cost = 1,7657
Epoch 36, Update 1600, Cost = 1,3730
Epoch 38, Update 1700, Cost = 1,5156
Epoch 40, Update 1800, Cost = 1,6843
Epoch 43, Update 1900, Cost = 1,4716
Epoch 45, Update 2000, Cost = 1,4910
Epoch 47, Update 2100, Cost = 1,6525
Epoch 49, Update 2200, Cost = 1,6935
Epoch 52, Update 2300, Cost = 1,3762
Epoch 54, Update 2400, Cost = 1,3785
Epoch 56, Update 2500, Cost = 1,6237
Epoch 59, Update 2600, Cost = 1,7075
Epoch 61, Update 2700, Cost = 1,3365
Epoch 63, Update 2800, Cost = 1,4869
Epoch 65, Update 2900, Cost = 1,6651
Epoch 68, Update 3000, Cost = 1,4588
Epoch 70, Update 3100, Cost = 1,4829
Epoch 72, Update 3200, Cost = 1,6460
Epoch 74, Update 3300, Cost = 1,6888
Epoch 77, Update 3400, Cost = 1,3737
Epoch 79, Update 3500, Cost = 1,3767
Epoch 81, Update 3600, Cost = 1,6225
Epoch 84, Update 3700, Cost = 1,7068
Epoch 86, Update 3800, Cost = 1,3361
Epoch 88, Update 3900, Cost = 1,4866
Epoch 90, Update 4000, Cost = 1,6649
Epoch 93, Update 4100, Cost = 1,4588
Epoch 95, Update 4200, Cost = 1,4828
Epoch 97, Update 4300, Cost = 1,6460
Epoch 99, Update 4400, Cost = 1,6888
Epoch 102, Update 4500, Cost = 1,3737
Epoch 104, Update 4600, Cost = 1,3767
Epoch 106, Update 4700, Cost = 1,6225
Epoch 109, Update 4800, Cost = 1,7068
Translations:
<s> Il cane è al </s>
<s> Il gatto è sul </s>
<s> Il gatto è sul davanzale. </s>
<s> Il gatto guarda l'uccello. </s>
<s> Dov'è il cane? </s>
<s> Chi ha un gatto? </s>
             */

            /*
             Epoch 2, Update 100, Cost = 11,8997
Translations:
<s> Il gatto il il </s>
<s> Il gatto il il il </s>
<s> Il gatto il il </s>
<s> Il gatto il il il </s>
<s> Il cane è </s>
<s> Il cane cane </s>
Epoch 2, Update 100, Cost = 13,7626
Epoch 4, Update 200, Cost = 11,0622
Epoch 6, Update 300, Cost = 10,1375
Epoch 9, Update 400, Cost = 8,3217
Epoch 11, Update 500, Cost = 8,4745
Translations:
<s> Il cane è in </s>
<s> Il gatto è sul </s>
<s> Il tuo gatto è vecchio. </s>
<s> Il gatto è sul il </s>
<s> Dov'è il il cane? </s>
<s> Lei guarda il film. </s>
Epoch 2, Update 100, Cost = 11,0049
Epoch 4, Update 200, Cost = 9,7593
Epoch 6, Update 300, Cost = 10,4610
Epoch 9, Update 400, Cost = 5,7702
Epoch 11, Update 500, Cost = 6,8415
Epoch 13, Update 600, Cost = 5,7751
Epoch 15, Update 700, Cost = 5,2561
Epoch 18, Update 800, Cost = 4,3719
Translations:
<s> Il cane è sul tavolo. </s>
<s> Il gatto è sul tavolo. </s>
<s> Il gatto gatto è curioso. </s>
<s> Il gatto e il cane. </s>
<s> È il gatto? </s>
<s> Hai un cane. </s>
Epoch 2, Update 100, Cost = 12,3263
Epoch 4, Update 200, Cost = 12,2330
Epoch 6, Update 300, Cost = 11,3670
Epoch 9, Update 400, Cost = 5,2984
Epoch 11, Update 500, Cost = 5,8542
Epoch 13, Update 600, Cost = 4,5111
Epoch 15, Update 700, Cost = 4,7521
Epoch 18, Update 800, Cost = 1,9055
Epoch 20, Update 900, Cost = 2,6597
Epoch 22, Update 1000, Cost = 2,3486
Epoch 24, Update 1100, Cost = 2,4555
Epoch 27, Update 1200, Cost = 1,4310
Epoch 29, Update 1300, Cost = 1,8070
Epoch 31, Update 1400, Cost = 2,0133
Epoch 34, Update 1500, Cost = 0,8679
Epoch 36, Update 1600, Cost = 1,6820
Epoch 38, Update 1700, Cost = 1,7340
Epoch 40, Update 1800, Cost = 1,9280
Epoch 43, Update 1900, Cost = 1,0099
Epoch 45, Update 2000, Cost = 1,7123
Epoch 47, Update 2100, Cost = 1,7178
Epoch 49, Update 2200, Cost = 1,9474
Epoch 52, Update 2300, Cost = 1,2074
Epoch 54, Update 2400, Cost = 1,6167
Translations:
<s> Il cane è in giardino. </s>
<s> Il gatto è sul tavolo. </s>
<s> Il gatto è sul letto. </s>
<s> Il gatto guarda al cane. </s>
<s> È mangia gatto? </s>
<s> Hai un cane. </s>
Epoch 2, Update 100, Cost = 11,3660
Epoch 4, Update 200, Cost = 10,0646
Epoch 6, Update 300, Cost = 11,0980
Epoch 9, Update 400, Cost = 5,9970
Epoch 11, Update 500, Cost = 7,2366
Epoch 13, Update 600, Cost = 5,5369
Epoch 15, Update 700, Cost = 5,0081
Epoch 18, Update 800, Cost = 3,6814
Epoch 20, Update 900, Cost = 3,6926
Epoch 22, Update 1000, Cost = 3,6236
Epoch 24, Update 1100, Cost = 3,3130
Epoch 27, Update 1200, Cost = 2,9736
Epoch 29, Update 1300, Cost = 2,6134
Epoch 31, Update 1400, Cost = 2,9427
Epoch 34, Update 1500, Cost = 2,1779
Epoch 36, Update 1600, Cost = 2,8964
Epoch 38, Update 1700, Cost = 2,7842
Epoch 40, Update 1800, Cost = 2,8163
Epoch 43, Update 1900, Cost = 2,4389
Epoch 45, Update 2000, Cost = 2,7630
Epoch 47, Update 2100, Cost = 2,9820
Epoch 49, Update 2200, Cost = 2,8625
Epoch 52, Update 2300, Cost = 2,6953
Epoch 54, Update 2400, Cost = 2,4334
Epoch 56, Update 2500, Cost = 2,8158
Epoch 59, Update 2600, Cost = 2,1023
Epoch 61, Update 2700, Cost = 2,8315
Epoch 63, Update 2800, Cost = 2,7399
Epoch 65, Update 2900, Cost = 2,7898
Translations:
<s> Il cane è al </s>
<s> Il gatto è sul tavolo. </s>
<s> Il gatto è sul davanzale. </s>
<s> Il gatto è il cane. </s>
<s> Dov'è il cane? </s>
<s> Hai un cane. </s>
Epoch 2, Update 100, Cost = 10,5466
Epoch 4, Update 200, Cost = 11,1326
Epoch 6, Update 300, Cost = 7,9523
Epoch 9, Update 400, Cost = 6,8900
Epoch 11, Update 500, Cost = 5,1055
Epoch 13, Update 600, Cost = 4,9235
Epoch 15, Update 700, Cost = 3,3998
Epoch 18, Update 800, Cost = 2,3996
Epoch 20, Update 900, Cost = 2,1495
Epoch 22, Update 1000, Cost = 2,1664
Epoch 24, Update 1100, Cost = 1,9635
Epoch 27, Update 1200, Cost = 1,6245
Epoch 29, Update 1300, Cost = 1,5655
Epoch 31, Update 1400, Cost = 1,7683
Epoch 34, Update 1500, Cost = 1,7723
Epoch 36, Update 1600, Cost = 1,4595
Epoch 38, Update 1700, Cost = 1,5606
Epoch 40, Update 1800, Cost = 1,7182
Epoch 43, Update 1900, Cost = 1,5985
Epoch 45, Update 2000, Cost = 1,5504
Epoch 47, Update 2100, Cost = 1,7103
Epoch 49, Update 2200, Cost = 1,6954
Epoch 52, Update 2300, Cost = 1,4730
Epoch 54, Update 2400, Cost = 1,4630
Epoch 56, Update 2500, Cost = 1,6781
Epoch 59, Update 2600, Cost = 1,7223
Epoch 61, Update 2700, Cost = 1,4242
Epoch 63, Update 2800, Cost = 1,5361
Epoch 65, Update 2900, Cost = 1,6993
Epoch 68, Update 3000, Cost = 1,5890
Epoch 70, Update 3100, Cost = 1,5427
Epoch 72, Update 3200, Cost = 1,7036
Epoch 74, Update 3300, Cost = 1,6915
Epoch 77, Update 3400, Cost = 1,4706
Epoch 79, Update 3500, Cost = 1,4615
Epoch 81, Update 3600, Cost = 1,6768
Epoch 84, Update 3700, Cost = 1,7217
Epoch 86, Update 3800, Cost = 1,4238
Translations:
<s> Il cane è sul pavimento. </s>
<s> Il gatto è sul davanzale. </s>
<s> Il gatto è sul davanzale. </s>
<s> Il gatto guarda l'uccello. </s>
<s> Dov'è il cane? </s>
<s> Lei dorme il gatto? </s>
Epoch 2, Update 100, Cost = 11,1015
Epoch 4, Update 200, Cost = 9,8248
Epoch 6, Update 300, Cost = 9,8518
Epoch 9, Update 400, Cost = 5,3633
Epoch 11, Update 500, Cost = 5,0749
Epoch 13, Update 600, Cost = 4,4040
Epoch 15, Update 700, Cost = 4,1061
Epoch 18, Update 800, Cost = 2,6181
Epoch 20, Update 900, Cost = 2,3654
Epoch 22, Update 1000, Cost = 2,6943
Epoch 24, Update 1100, Cost = 2,5831
Epoch 27, Update 1200, Cost = 1,9378
Epoch 29, Update 1300, Cost = 1,6857
Epoch 31, Update 1400, Cost = 2,2066
Epoch 34, Update 1500, Cost = 1,8932
Epoch 36, Update 1600, Cost = 1,6832
Epoch 38, Update 1700, Cost = 1,8915
Epoch 40, Update 1800, Cost = 2,1489
Epoch 43, Update 1900, Cost = 1,8178
Epoch 45, Update 2000, Cost = 1,7738
Epoch 47, Update 2100, Cost = 2,1876
Epoch 49, Update 2200, Cost = 2,2151
Epoch 52, Update 2300, Cost = 1,7660
Epoch 54, Update 2400, Cost = 1,5736
Epoch 56, Update 2500, Cost = 2,1029
Epoch 59, Update 2600, Cost = 1,8481
Epoch 61, Update 2700, Cost = 1,6467
Epoch 63, Update 2800, Cost = 1,8597
Epoch 65, Update 2900, Cost = 2,1270
Epoch 68, Update 3000, Cost = 1,8063
Epoch 70, Update 3100, Cost = 1,7660
Epoch 72, Update 3200, Cost = 2,1801
Epoch 74, Update 3300, Cost = 2,2095
Epoch 77, Update 3400, Cost = 1,7635
Epoch 79, Update 3500, Cost = 1,5720
Epoch 81, Update 3600, Cost = 2,1015
Epoch 84, Update 3700, Cost = 1,8475
Epoch 86, Update 3800, Cost = 1,6463
Epoch 88, Update 3900, Cost = 1,8594
Epoch 90, Update 4000, Cost = 2,1268
Epoch 93, Update 4100, Cost = 1,8063
Epoch 95, Update 4200, Cost = 1,7659
Epoch 97, Update 4300, Cost = 2,1801
Epoch 99, Update 4400, Cost = 2,2095
Epoch 102, Update 4500, Cost = 1,7635
Epoch 104, Update 4600, Cost = 1,5720
Epoch 106, Update 4700, Cost = 2,1015
Epoch 109, Update 4800, Cost = 1,8475
Translations:
<s> Il cane è il il </s>
<s> Il gatto mangia pesce. </s>
<s> Il gatto dorme dorme davanzale. </s>
<s> Il gatto guarda l'uccello. </s>
<s> Dov'è il cane? </s>
<s> Hai un cane. </s>
             */

            Console.ReadLine();
        }
    }
}
