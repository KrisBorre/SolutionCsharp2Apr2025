using Seq2SeqSharp;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Enums;
using Seq2SeqSharp.LearningRate;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;

// ChatGPT, modified
// Application uses 1500 MegaByte of RAM. 
// 3 layers
namespace ConsoleDutchItalianSeq2Seq17Jul2025
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
                    EncoderLayerDepth = 3,
                    DecoderLayerDepth = 3,
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
Epoch 2, Update 100, Cost = 12,5928
Translations:
<s> Il gatto è il </s>
<s> Il gatto è il </s>
<s> Il il il </s>
<s> Il gatto è il </s>
<s> Il il gatto? </s>
<s> Il il gatto? </s>
Epoch 2, Update 100, Cost = 11,2806
Epoch 4, Update 200, Cost = 9,3638
Epoch 6, Update 300, Cost = 7,6814
Epoch 9, Update 400, Cost = 3,8172
Epoch 11, Update 500, Cost = 3,1731
Translations:
<s> Il cane è in </s>
<s> Il gatto è sul </s>
<s> Il gatto sul sul </s>
<s> Il gatto guarda l'uccello. </s>
<s> Dov'è un gatti. </s>
<s> Chi ha un </s>
Epoch 2, Update 100, Cost = 12,1118
Epoch 4, Update 200, Cost = 10,5467
Epoch 6, Update 300, Cost = 8,2858
Epoch 9, Update 400, Cost = 6,1562
Epoch 11, Update 500, Cost = 4,3341
Epoch 13, Update 600, Cost = 2,7019
Epoch 15, Update 700, Cost = 1,7087
Epoch 18, Update 800, Cost = 1,0933
Translations:
<s> Il cane è in giardino. </s>
<s> Il gatto è sul tavolo. </s>
<s> Il gatto gatto sul letto. </s>
<s> Il gatto guarda l'uccello. </s>
<s> Dov'è il cane? </s>
<s> Lei è sulla sedia. </s>
Epoch 2, Update 100, Cost = 10,9499
Epoch 4, Update 200, Cost = 7,3383
Epoch 6, Update 300, Cost = 5,4401
Epoch 9, Update 400, Cost = 2,4850
Epoch 11, Update 500, Cost = 1,4675
Epoch 13, Update 600, Cost = 0,9589
Epoch 15, Update 700, Cost = 0,7086
Epoch 18, Update 800, Cost = 0,2186
Epoch 20, Update 900, Cost = 0,2540
Epoch 22, Update 1000, Cost = 0,2050
Epoch 24, Update 1100, Cost = 0,2204
Epoch 27, Update 1200, Cost = 0,1231
Epoch 29, Update 1300, Cost = 0,1244
Epoch 31, Update 1400, Cost = 0,1585
Epoch 34, Update 1500, Cost = 0,0690
Epoch 36, Update 1600, Cost = 0,1147
Epoch 38, Update 1700, Cost = 0,1154
Epoch 40, Update 1800, Cost = 0,1432
Epoch 43, Update 1900, Cost = 0,0652
Epoch 45, Update 2000, Cost = 0,1114
Epoch 47, Update 2100, Cost = 0,1123
Epoch 49, Update 2200, Cost = 0,1471
Epoch 52, Update 2300, Cost = 0,0935
Epoch 54, Update 2400, Cost = 0,1029
Translations:
<s> Il cane è in giardino. </s>
<s> Il gatto è sul </s>
<s> Il gatto dorme sul davanzale. </s>
<s> Il gatto è accanto cane. </s>
<s> Quando il gatto? </s>
<s> Hai un cane. </s>
Epoch 2, Update 100, Cost = 11,0777
Epoch 4, Update 200, Cost = 9,8533
Epoch 6, Update 300, Cost = 7,8103
Epoch 9, Update 400, Cost = 3,8774
Epoch 11, Update 500, Cost = 3,8082
Epoch 13, Update 600, Cost = 2,2036
Epoch 15, Update 700, Cost = 1,3526
Epoch 18, Update 800, Cost = 0,4572
Epoch 20, Update 900, Cost = 0,6615
Epoch 22, Update 1000, Cost = 0,5528
Epoch 24, Update 1100, Cost = 0,5116
Epoch 27, Update 1200, Cost = 0,2848
Epoch 29, Update 1300, Cost = 0,3581
Epoch 31, Update 1400, Cost = 0,3530
Epoch 34, Update 1500, Cost = 0,2057
Epoch 36, Update 1600, Cost = 0,3157
Epoch 38, Update 1700, Cost = 0,3154
Epoch 40, Update 1800, Cost = 0,3093
Epoch 43, Update 1900, Cost = 0,1740
Epoch 45, Update 2000, Cost = 0,2933
Epoch 47, Update 2100, Cost = 0,2950
Epoch 49, Update 2200, Cost = 0,3367
Epoch 52, Update 2300, Cost = 0,2151
Epoch 54, Update 2400, Cost = 0,2885
Epoch 56, Update 2500, Cost = 0,3062
Epoch 59, Update 2600, Cost = 0,1896
Epoch 61, Update 2700, Cost = 0,2957
Epoch 63, Update 2800, Cost = 0,3004
Epoch 65, Update 2900, Cost = 0,2998
Translations:
<s> Il cane è in giardino. </s>
<s> Il gatto è sul tavolo. </s>
<s> Il gatto dorme sul sul </s>
<s> Il gatto guarda l'uccello. </s>
<s> Ha il gatto? gatto? </s>
<s> Chi un un </s>
Epoch 2, Update 100, Cost = 11,6966
Epoch 4, Update 200, Cost = 8,7819
Epoch 6, Update 300, Cost = 5,4757
Epoch 9, Update 400, Cost = 1,6465
Epoch 11, Update 500, Cost = 1,8759
Epoch 13, Update 600, Cost = 0,8117
Epoch 15, Update 700, Cost = 0,8062
Epoch 18, Update 800, Cost = 0,2244
Epoch 20, Update 900, Cost = 0,2695
Epoch 22, Update 1000, Cost = 0,4435
Epoch 24, Update 1100, Cost = 0,4260
Epoch 27, Update 1200, Cost = 0,1666
Epoch 29, Update 1300, Cost = 0,1857
Epoch 31, Update 1400, Cost = 0,3966
Epoch 34, Update 1500, Cost = 0,0972
Epoch 36, Update 1600, Cost = 0,1741
Epoch 38, Update 1700, Cost = 0,2344
Epoch 40, Update 1800, Cost = 0,3603
Epoch 43, Update 1900, Cost = 0,1057
Epoch 45, Update 2000, Cost = 0,1641
Epoch 47, Update 2100, Cost = 0,3556
Epoch 49, Update 2200, Cost = 0,3575
Epoch 52, Update 2300, Cost = 0,1425
Epoch 54, Update 2400, Cost = 0,1685
Epoch 56, Update 2500, Cost = 0,3772
Epoch 59, Update 2600, Cost = 0,0929
Epoch 61, Update 2700, Cost = 0,1686
Epoch 63, Update 2800, Cost = 0,2301
Epoch 65, Update 2900, Cost = 0,3563
Epoch 68, Update 3000, Cost = 0,1046
Epoch 70, Update 3100, Cost = 0,1629
Epoch 72, Update 3200, Cost = 0,3546
Epoch 74, Update 3300, Cost = 0,3566
Epoch 77, Update 3400, Cost = 0,1422
Epoch 79, Update 3500, Cost = 0,1683
Epoch 81, Update 3600, Cost = 0,3770
Epoch 84, Update 3700, Cost = 0,0929
Epoch 86, Update 3800, Cost = 0,1685
Translations:
<s> Il cane è in macchina. </s>
<s> Il gatto è sul tavolo. </s>
<s> Il gatto dorme sul davanzale. </s>
<s> Il gatto è cane. cane. </s>
<s> Dove il il </s>
<s> Hai un cane. </s>
Epoch 2, Update 100, Cost = 12,1657
Epoch 4, Update 200, Cost = 10,8286
Epoch 6, Update 300, Cost = 7,4822
Epoch 9, Update 400, Cost = 4,0196
Epoch 11, Update 500, Cost = 2,4402
Epoch 13, Update 600, Cost = 1,4049
Epoch 15, Update 700, Cost = 0,9906
Epoch 18, Update 800, Cost = 0,3621
Epoch 20, Update 900, Cost = 0,4254
Epoch 22, Update 1000, Cost = 0,3207
Epoch 24, Update 1100, Cost = 0,3491
Epoch 27, Update 1200, Cost = 0,1872
Epoch 29, Update 1300, Cost = 0,2272
Epoch 31, Update 1400, Cost = 0,2608
Epoch 34, Update 1500, Cost = 0,1113
Epoch 36, Update 1600, Cost = 0,2163
Epoch 38, Update 1700, Cost = 0,1979
Epoch 40, Update 1800, Cost = 0,2348
Epoch 43, Update 1900, Cost = 0,1049
Epoch 45, Update 2000, Cost = 0,1996
Epoch 47, Update 2100, Cost = 0,1836
Epoch 49, Update 2200, Cost = 0,2377
Epoch 52, Update 2300, Cost = 0,1429
Epoch 54, Update 2400, Cost = 0,1888
Epoch 56, Update 2500, Cost = 0,2306
Epoch 59, Update 2600, Cost = 0,1010
Epoch 61, Update 2700, Cost = 0,2042
Epoch 63, Update 2800, Cost = 0,1900
Epoch 65, Update 2900, Cost = 0,2285
Epoch 68, Update 3000, Cost = 0,1028
Epoch 70, Update 3100, Cost = 0,1971
Epoch 72, Update 3200, Cost = 0,1819
Epoch 74, Update 3300, Cost = 0,2362
Epoch 77, Update 3400, Cost = 0,1424
Epoch 79, Update 3500, Cost = 0,1883
Epoch 81, Update 3600, Cost = 0,2302
Epoch 84, Update 3700, Cost = 0,1009
Epoch 86, Update 3800, Cost = 0,2041
Epoch 88, Update 3900, Cost = 0,1899
Epoch 90, Update 4000, Cost = 0,2284
Epoch 93, Update 4100, Cost = 0,1028
Epoch 95, Update 4200, Cost = 0,1971
Epoch 97, Update 4300, Cost = 0,1819
Epoch 99, Update 4400, Cost = 0,2362
Epoch 102, Update 4500, Cost = 0,1424
Epoch 104, Update 4600, Cost = 0,1883
Epoch 106, Update 4700, Cost = 0,2302
Epoch 109, Update 4800, Cost = 0,1009
Translations:
<s> Il cane è in giardino. </s>
<s> Il gatto è il tavolo. </s>
<s> Il gatto è sul </s>
<s> Il gatto guarda l'uccello. </s>
<s> Dove dorme gatto? </s>
<s> Hai un cane. </s>
Epoch 2, Update 100, Cost = 11,6068
Epoch 4, Update 200, Cost = 8,4591
Epoch 6, Update 300, Cost = 5,8982
Epoch 9, Update 400, Cost = 2,5153
Epoch 11, Update 500, Cost = 1,7652
Epoch 13, Update 600, Cost = 1,3160
Epoch 15, Update 700, Cost = 0,8304
Epoch 18, Update 800, Cost = 0,2899
Epoch 20, Update 900, Cost = 0,3338
Epoch 22, Update 1000, Cost = 0,3920
Epoch 24, Update 1100, Cost = 0,3799
Epoch 27, Update 1200, Cost = 0,1844
Epoch 29, Update 1300, Cost = 0,2119
Epoch 31, Update 1400, Cost = 0,3199
Epoch 34, Update 1500, Cost = 0,1105
Epoch 36, Update 1600, Cost = 0,1858
Epoch 38, Update 1700, Cost = 0,2062
Epoch 40, Update 1800, Cost = 0,2982
Epoch 43, Update 1900, Cost = 0,1152
Epoch 45, Update 2000, Cost = 0,1863
Epoch 47, Update 2100, Cost = 0,2832
Epoch 49, Update 2200, Cost = 0,3001
Epoch 52, Update 2300, Cost = 0,1535
Epoch 54, Update 2400, Cost = 0,1884
Epoch 56, Update 2500, Cost = 0,2986
Epoch 59, Update 2600, Cost = 0,1043
Epoch 61, Update 2700, Cost = 0,1792
Epoch 63, Update 2800, Cost = 0,2009
Epoch 65, Update 2900, Cost = 0,2936
Epoch 68, Update 3000, Cost = 0,1138
Epoch 70, Update 3100, Cost = 0,1847
Epoch 72, Update 3200, Cost = 0,2820
Epoch 74, Update 3300, Cost = 0,2991
Epoch 77, Update 3400, Cost = 0,1532
Epoch 79, Update 3500, Cost = 0,1881
Epoch 81, Update 3600, Cost = 0,2984
Epoch 84, Update 3700, Cost = 0,1042
Epoch 86, Update 3800, Cost = 0,1791
Epoch 88, Update 3900, Cost = 0,2009
Epoch 90, Update 4000, Cost = 0,2935
Epoch 93, Update 4100, Cost = 0,1138
Epoch 95, Update 4200, Cost = 0,1847
Epoch 97, Update 4300, Cost = 0,2820
Epoch 99, Update 4400, Cost = 0,2991
Epoch 102, Update 4500, Cost = 0,1532
Epoch 104, Update 4600, Cost = 0,1881
Epoch 106, Update 4700, Cost = 0,2984
Epoch 109, Update 4800, Cost = 0,1042
Epoch 111, Update 4900, Cost = 0,1791
Epoch 113, Update 5000, Cost = 0,2009
Epoch 115, Update 5100, Cost = 0,2935
Epoch 118, Update 5200, Cost = 0,1138
Epoch 120, Update 5300, Cost = 0,1847
Epoch 122, Update 5400, Cost = 0,2820
Epoch 124, Update 5500, Cost = 0,2991
Epoch 127, Update 5600, Cost = 0,1532
Epoch 129, Update 5700, Cost = 0,1881
Epoch 131, Update 5800, Cost = 0,2984
Epoch 134, Update 5900, Cost = 0,1042
Epoch 136, Update 6000, Cost = 0,1791
Epoch 138, Update 6100, Cost = 0,2009
Epoch 140, Update 6200, Cost = 0,2935
Epoch 143, Update 6300, Cost = 0,1138
Epoch 145, Update 6400, Cost = 0,1847
Epoch 147, Update 6500, Cost = 0,2820
Epoch 149, Update 6600, Cost = 0,2991
Epoch 152, Update 6700, Cost = 0,1532
Epoch 154, Update 6800, Cost = 0,1881
Epoch 156, Update 6900, Cost = 0,2984
Epoch 159, Update 7000, Cost = 0,1042
Epoch 161, Update 7100, Cost = 0,1791
Epoch 163, Update 7200, Cost = 0,2009
Epoch 165, Update 7300, Cost = 0,2935
Epoch 168, Update 7400, Cost = 0,1138
Epoch 170, Update 7500, Cost = 0,1847
Epoch 172, Update 7600, Cost = 0,2820
Epoch 174, Update 7700, Cost = 0,2991
Epoch 177, Update 7800, Cost = 0,1532
Epoch 179, Update 7900, Cost = 0,1881
Epoch 181, Update 8000, Cost = 0,2984
Epoch 184, Update 8100, Cost = 0,1042
Epoch 186, Update 8200, Cost = 0,1791
Epoch 188, Update 8300, Cost = 0,2009
Epoch 190, Update 8400, Cost = 0,2935
Epoch 193, Update 8500, Cost = 0,1138
Epoch 195, Update 8600, Cost = 0,1847
Epoch 197, Update 8700, Cost = 0,2820
Epoch 199, Update 8800, Cost = 0,2991
Translations:
<s> Il cane è in giardino. </s>
<s> Il gatto è sul </s>
<s> Il gatto è sul </s>
<s> Il gatto guarda l'uccello. </s>
<s> Dov'è gatto è </s>
<s> Chi ha un gatto? </s>
       */


            /*
            Epoch 2, Update 100, Cost = 12,1555
Translations:
<s> Il cane è è </s>
<s> Il cane è è </s>
<s> Il il il </s>
<s> Il cane è il </s>
<s> È il </s>
<s> un un gatto? </s>
Epoch 2, Update 100, Cost = 12,0886
Epoch 4, Update 200, Cost = 12,4042
Epoch 6, Update 300, Cost = 11,1967
Epoch 9, Update 400, Cost = 6,0465
Epoch 11, Update 500, Cost = 5,1408
Translations:
<s> Il cane è in in </s>
<s> Il gatto è sul tavolo. </s>
<s> Il gatto è sul sul </s>
<s> Il gatto è il cane. </s>
<s> Ha gatto gatto? gatto? </s>
<s> Hai un un </s>
Epoch 2, Update 100, Cost = 12,2801
Epoch 4, Update 200, Cost = 14,8090
Epoch 6, Update 300, Cost = 13,7231
Epoch 9, Update 400, Cost = 8,6849
Epoch 11, Update 500, Cost = 9,4936
Epoch 13, Update 600, Cost = 8,3934
Epoch 15, Update 700, Cost = 7,2440
Epoch 18, Update 800, Cost = 5,2287
Translations:
<s> Il cane è in sole. </s>
<s> Il gatto mangia sul </s>
<s> Il gatto è sul sedia. </s>
<s> Il gatto guarda il </s>
<s> Dov'è il cane? </s>
<s> Vuoi un un gatto? </s>
Epoch 2, Update 100, Cost = 11,1732
Epoch 4, Update 200, Cost = 11,6632
Epoch 6, Update 300, Cost = 9,9458
Epoch 9, Update 400, Cost = 6,5602
Epoch 11, Update 500, Cost = 6,5852
Epoch 13, Update 600, Cost = 5,0314
Epoch 15, Update 700, Cost = 4,1246
Epoch 18, Update 800, Cost = 3,7994
Epoch 20, Update 900, Cost = 3,0169
Epoch 22, Update 1000, Cost = 2,9069
Epoch 24, Update 1100, Cost = 2,7590
Epoch 27, Update 1200, Cost = 2,9493
Epoch 29, Update 1300, Cost = 2,0428
Epoch 31, Update 1400, Cost = 2,3501
Epoch 34, Update 1500, Cost = 2,8970
Epoch 36, Update 1600, Cost = 2,4625
Epoch 38, Update 1700, Cost = 2,1411
Epoch 40, Update 1800, Cost = 2,1669
Epoch 43, Update 1900, Cost = 2,5827
Epoch 45, Update 2000, Cost = 2,1616
Epoch 47, Update 2100, Cost = 2,2960
Epoch 49, Update 2200, Cost = 2,3232
Epoch 52, Update 2300, Cost = 2,6270
Epoch 54, Update 2400, Cost = 1,8799
Translations:
<s> Il cane è in pavimento. </s>
<s> Il gatto è sul </s>
<s> Il gatto gatto è al </s>
<s> Il gatto è il </s>
<s> Dove dorme gatto? gatto? </s>
<s> Scrivi un cane. </s>
Epoch 2, Update 100, Cost = 13,2197
Epoch 4, Update 200, Cost = 14,9738
Epoch 6, Update 300, Cost = 14,8165
Epoch 9, Update 400, Cost = 10,0151
Epoch 11, Update 500, Cost = 10,8695
Epoch 13, Update 600, Cost = 10,4564
Epoch 15, Update 700, Cost = 9,4755
Epoch 18, Update 800, Cost = 6,9382
Epoch 20, Update 900, Cost = 7,7112
Epoch 22, Update 1000, Cost = 7,3231
Epoch 24, Update 1100, Cost = 7,1062
Epoch 27, Update 1200, Cost = 6,6458
Epoch 29, Update 1300, Cost = 5,8990
Epoch 31, Update 1400, Cost = 6,2490
Epoch 34, Update 1500, Cost = 5,2004
Epoch 36, Update 1600, Cost = 5,9681
Epoch 38, Update 1700, Cost = 5,8999
Epoch 40, Update 1800, Cost = 5,9979
Epoch 43, Update 1900, Cost = 5,4671
Epoch 45, Update 2000, Cost = 5,9336
Epoch 47, Update 2100, Cost = 6,0882
Epoch 49, Update 2200, Cost = 6,2197
Epoch 52, Update 2300, Cost = 6,1455
Epoch 54, Update 2400, Cost = 5,5031
Epoch 56, Update 2500, Cost = 5,9706
Epoch 59, Update 2600, Cost = 5,1025
Epoch 61, Update 2700, Cost = 5,8403
Epoch 63, Update 2800, Cost = 5,8117
Epoch 65, Update 2900, Cost = 5,9388
Translations:
<s> Il gatto è sul tavolo. </s>
<s> Il gatto è sul il </s>
<s> Il gatto è sul davanzale. </s>
<s> Il gatto guarda il </s>
<s> Dov'è il cane? </s>
<s> Lei guarda il film. </s>
Epoch 2, Update 100, Cost = 12,9307
Epoch 4, Update 200, Cost = 12,0911
Epoch 6, Update 300, Cost = 10,6608
Epoch 9, Update 400, Cost = 4,9109
Epoch 11, Update 500, Cost = 5,0325
Epoch 13, Update 600, Cost = 4,2982
Epoch 15, Update 700, Cost = 3,4231
Epoch 18, Update 800, Cost = 1,9261
Epoch 20, Update 900, Cost = 1,8594
Epoch 22, Update 1000, Cost = 1,8623
Epoch 24, Update 1100, Cost = 1,8623
Epoch 27, Update 1200, Cost = 1,3148
Epoch 29, Update 1300, Cost = 1,1984
Epoch 31, Update 1400, Cost = 1,4597
Epoch 34, Update 1500, Cost = 1,1069
Epoch 36, Update 1600, Cost = 1,1541
Epoch 38, Update 1700, Cost = 1,1894
Epoch 40, Update 1800, Cost = 1,3979
Epoch 43, Update 1900, Cost = 0,9889
Epoch 45, Update 2000, Cost = 1,1561
Epoch 47, Update 2100, Cost = 1,3411
Epoch 49, Update 2200, Cost = 1,4848
Epoch 52, Update 2300, Cost = 1,1082
Epoch 54, Update 2400, Cost = 1,0749
Epoch 56, Update 2500, Cost = 1,3553
Epoch 59, Update 2600, Cost = 1,0439
Epoch 61, Update 2700, Cost = 1,1114
Epoch 63, Update 2800, Cost = 1,1616
Epoch 65, Update 2900, Cost = 1,3749
Epoch 68, Update 3000, Cost = 0,9764
Epoch 70, Update 3100, Cost = 1,1469
Epoch 72, Update 3200, Cost = 1,3343
Epoch 74, Update 3300, Cost = 1,4795
Epoch 77, Update 3400, Cost = 1,1055
Epoch 79, Update 3500, Cost = 1,0732
Epoch 81, Update 3600, Cost = 1,3540
Epoch 84, Update 3700, Cost = 1,0431
Epoch 86, Update 3800, Cost = 1,1110
Translations:
<s> Il cane è in giardino. </s>
<s> Il gatto è sul </s>
<s> Il gatto dorme sul </s>
<s> Il gatto guarda l'uccello. </s>
<s> Il cane è </s>
<s> Chi ha un gatto? </s>
Epoch 2, Update 100, Cost = 11,8312
Epoch 4, Update 200, Cost = 12,1398
Epoch 6, Update 300, Cost = 11,4958
Epoch 9, Update 400, Cost = 6,4621
Epoch 11, Update 500, Cost = 6,6411
Epoch 13, Update 600, Cost = 5,5926
Epoch 15, Update 700, Cost = 4,4966
Epoch 18, Update 800, Cost = 3,4298
Epoch 20, Update 900, Cost = 3,1288
Epoch 22, Update 1000, Cost = 2,8373
Epoch 24, Update 1100, Cost = 2,6838
Epoch 27, Update 1200, Cost = 2,6188
Epoch 29, Update 1300, Cost = 2,1477
Epoch 31, Update 1400, Cost = 2,3118
Epoch 34, Update 1500, Cost = 2,5249
Epoch 36, Update 1600, Cost = 2,2340
Epoch 38, Update 1700, Cost = 2,0650
Epoch 40, Update 1800, Cost = 2,2092
Epoch 43, Update 1900, Cost = 2,3611
Epoch 45, Update 2000, Cost = 2,1474
Epoch 47, Update 2100, Cost = 2,1778
Epoch 49, Update 2200, Cost = 2,2043
Epoch 52, Update 2300, Cost = 2,3085
Epoch 54, Update 2400, Cost = 1,9673
Epoch 56, Update 2500, Cost = 2,1752
Epoch 59, Update 2600, Cost = 2,4513
Epoch 61, Update 2700, Cost = 2,1706
Epoch 63, Update 2800, Cost = 2,0230
Epoch 65, Update 2900, Cost = 2,1798
Epoch 68, Update 3000, Cost = 2,3431
Epoch 70, Update 3100, Cost = 2,1337
Epoch 72, Update 3200, Cost = 2,1682
Epoch 74, Update 3300, Cost = 2,1979
Epoch 77, Update 3400, Cost = 2,3044
Epoch 79, Update 3500, Cost = 1,9647
Epoch 81, Update 3600, Cost = 2,1734
Epoch 84, Update 3700, Cost = 2,4504
Epoch 86, Update 3800, Cost = 2,1699
Epoch 88, Update 3900, Cost = 2,0226
Epoch 90, Update 4000, Cost = 2,1796
Epoch 93, Update 4100, Cost = 2,3430
Epoch 95, Update 4200, Cost = 2,1336
Epoch 97, Update 4300, Cost = 2,1682
Epoch 99, Update 4400, Cost = 2,1979
Epoch 102, Update 4500, Cost = 2,3043
Epoch 104, Update 4600, Cost = 1,9647
Epoch 106, Update 4700, Cost = 2,1734
Epoch 109, Update 4800, Cost = 2,4504
Translations:
<s> Il cane è in giardino. </s>
<s> Il gatto è sul davanzale. </s>
<s> Il gatto dorme sul davanzale. </s>
<s> Il gatto guarda l'uccello. </s>
<s> Lui il il gatto? </s>
<s> Chi un un </s>
             */


            /*
             Epoch 2, Update 100, Cost = 11,4916
Translations:
<s> Il cane cane è </s>
<s> Il gatto gatto </s>
<s> Lui il </s>
<s> Il gatto gatto </s>
<s> </s>
<s> </s>
Epoch 2, Update 100, Cost = 11,2684
Epoch 4, Update 200, Cost = 11,8079
Epoch 6, Update 300, Cost = 10,3950
Epoch 9, Update 400, Cost = 6,2299
Epoch 11, Update 500, Cost = 7,3076
Translations:
<s> Il cane è il </s>
<s> Il gatto è sul sul </s>
<s> Dormo sul divano. </s>
<s> Il gatto è al al </s>
<s> È il gatto? gatto? </s>
<s> Chi ha un gatto? </s>
Epoch 2, Update 100, Cost = 11,2497
Epoch 4, Update 200, Cost = 11,2914
Epoch 6, Update 300, Cost = 9,6453
Epoch 9, Update 400, Cost = 5,5862
Epoch 11, Update 500, Cost = 4,4650
Epoch 13, Update 600, Cost = 3,6604
Epoch 15, Update 700, Cost = 2,9729
Epoch 18, Update 800, Cost = 2,3195
Translations:
<s> Il cane è in giardino. </s>
<s> Il gatto è sul </s>
<s> Lei sul sulla sedia. </s>
<s> Il gatto guarda l'uccello. </s>
<s> Dove dorme il gatto? </s>
<s> Hai un cane. </s>
Epoch 2, Update 100, Cost = 12,2960
Epoch 4, Update 200, Cost = 13,2312
Epoch 6, Update 300, Cost = 11,7223
Epoch 9, Update 400, Cost = 6,9850
Epoch 11, Update 500, Cost = 7,7674
Epoch 13, Update 600, Cost = 6,3044
Epoch 15, Update 700, Cost = 5,5641
Epoch 18, Update 800, Cost = 3,6617
Epoch 20, Update 900, Cost = 3,7855
Epoch 22, Update 1000, Cost = 3,5178
Epoch 24, Update 1100, Cost = 3,7486
Epoch 27, Update 1200, Cost = 2,6537
Epoch 29, Update 1300, Cost = 3,0058
Epoch 31, Update 1400, Cost = 3,3801
Epoch 34, Update 1500, Cost = 2,7651
Epoch 36, Update 1600, Cost = 2,7768
Epoch 38, Update 1700, Cost = 2,8903
Epoch 40, Update 1800, Cost = 3,2681
Epoch 43, Update 1900, Cost = 2,3852
Epoch 45, Update 2000, Cost = 2,7976
Epoch 47, Update 2100, Cost = 2,8203
Epoch 49, Update 2200, Cost = 3,2339
Epoch 52, Update 2300, Cost = 2,3220
Epoch 54, Update 2400, Cost = 2,8004
Translations:
<s> Il cane è in cucina. </s>
<s> Il gatto è sul </s>
<s> Il gatto è sul </s>
<s> Il gatto guarda l'uccello. </s>
<s> Lui il gatto? </s>
<s> Lui un un gatto? </s>
Epoch 2, Update 100, Cost = 11,4810
Epoch 4, Update 200, Cost = 10,5621
Epoch 6, Update 300, Cost = 9,1084
Epoch 9, Update 400, Cost = 4,5408
Epoch 11, Update 500, Cost = 3,0213
Epoch 13, Update 600, Cost = 2,6409
Epoch 15, Update 700, Cost = 1,9841
Epoch 18, Update 800, Cost = 1,5930
Epoch 20, Update 900, Cost = 1,1310
Epoch 22, Update 1000, Cost = 1,1744
Epoch 24, Update 1100, Cost = 1,0744
Epoch 27, Update 1200, Cost = 0,8765
Epoch 29, Update 1300, Cost = 0,7222
Epoch 31, Update 1400, Cost = 0,9307
Epoch 34, Update 1500, Cost = 0,7797
Epoch 36, Update 1600, Cost = 0,7455
Epoch 38, Update 1700, Cost = 0,7629
Epoch 40, Update 1800, Cost = 0,8568
Epoch 43, Update 1900, Cost = 0,7917
Epoch 45, Update 2000, Cost = 0,6929
Epoch 47, Update 2100, Cost = 0,8673
Epoch 49, Update 2200, Cost = 0,8713
Epoch 52, Update 2300, Cost = 0,7381
Epoch 54, Update 2400, Cost = 0,6431
Epoch 56, Update 2500, Cost = 0,8684
Epoch 59, Update 2600, Cost = 0,7386
Epoch 61, Update 2700, Cost = 0,7192
Epoch 63, Update 2800, Cost = 0,7442
Epoch 65, Update 2900, Cost = 0,8439
Translations:
<s> Il cane è in giardino. </s>
<s> Il gatto è sul davanzale. </s>
<s> Il gatto dorme sul davanzale. </s>
<s> Lei gatto il cane. cane. </s>
<s> È dorme il gatto? </s>
<s> Hai un cane. </s>
Epoch 2, Update 100, Cost = 12,6684
Epoch 4, Update 200, Cost = 12,7542
Epoch 6, Update 300, Cost = 9,9417
Epoch 9, Update 400, Cost = 4,1852
Epoch 11, Update 500, Cost = 3,8878
Epoch 13, Update 600, Cost = 2,5524
Epoch 15, Update 700, Cost = 2,1753
Epoch 18, Update 800, Cost = 0,8289
Epoch 20, Update 900, Cost = 0,9329
Epoch 22, Update 1000, Cost = 0,8043
Epoch 24, Update 1100, Cost = 1,0156
Epoch 27, Update 1200, Cost = 0,5133
Epoch 29, Update 1300, Cost = 0,5983
Epoch 31, Update 1400, Cost = 0,8370
Epoch 34, Update 1500, Cost = 0,3677
Epoch 36, Update 1600, Cost = 0,5328
Epoch 38, Update 1700, Cost = 0,5562
Epoch 40, Update 1800, Cost = 0,8043
Epoch 43, Update 1900, Cost = 0,3439
Epoch 45, Update 2000, Cost = 0,5607
Epoch 47, Update 2100, Cost = 0,5450
Epoch 49, Update 2200, Cost = 0,7976
Epoch 52, Update 2300, Cost = 0,4315
Epoch 54, Update 2400, Cost = 0,5311
Epoch 56, Update 2500, Cost = 0,7707
Epoch 59, Update 2600, Cost = 0,3443
Epoch 61, Update 2700, Cost = 0,5120
Epoch 63, Update 2800, Cost = 0,5416
Epoch 65, Update 2900, Cost = 0,7895
Epoch 68, Update 3000, Cost = 0,3392
Epoch 70, Update 3100, Cost = 0,5559
Epoch 72, Update 3200, Cost = 0,5418
Epoch 74, Update 3300, Cost = 0,7945
Epoch 77, Update 3400, Cost = 0,4303
Epoch 79, Update 3500, Cost = 0,5302
Epoch 81, Update 3600, Cost = 0,7699
Epoch 84, Update 3700, Cost = 0,3440
Epoch 86, Update 3800, Cost = 0,5117
Translations:
<s> Il cane è in giardino. </s>
<s> Il gatto è sul tavolo. </s>
<s> Il gatto è sul </s>
<s> Il gatto è al cane. </s>
<s> Quando dorme il gatto? </s>
<s> Chi ha un gatto? </s>
Epoch 2, Update 100, Cost = 12,6217
Epoch 4, Update 200, Cost = 12,1718
Epoch 6, Update 300, Cost = 10,3790
Epoch 9, Update 400, Cost = 6,1069
Epoch 11, Update 500, Cost = 5,6353
Epoch 13, Update 600, Cost = 4,5665
Epoch 15, Update 700, Cost = 3,6143
Epoch 18, Update 800, Cost = 3,1066
Epoch 20, Update 900, Cost = 2,3909
Epoch 22, Update 1000, Cost = 2,3837
Epoch 24, Update 1100, Cost = 2,1880
Epoch 27, Update 1200, Cost = 2,1947
Epoch 29, Update 1300, Cost = 1,5889
Epoch 31, Update 1400, Cost = 1,8727
Epoch 34, Update 1500, Cost = 1,9448
Epoch 36, Update 1600, Cost = 1,8214
Epoch 38, Update 1700, Cost = 1,6608
Epoch 40, Update 1800, Cost = 1,7650
Epoch 43, Update 1900, Cost = 1,9185
Epoch 45, Update 2000, Cost = 1,6033
Epoch 47, Update 2100, Cost = 1,8242
Epoch 49, Update 2200, Cost = 1,7963
Epoch 52, Update 2300, Cost = 1,9208
Epoch 54, Update 2400, Cost = 1,4484
Epoch 56, Update 2500, Cost = 1,7635
Epoch 59, Update 2600, Cost = 1,8761
Epoch 61, Update 2700, Cost = 1,7671
Epoch 63, Update 2800, Cost = 1,6273
Epoch 65, Update 2900, Cost = 1,7418
Epoch 68, Update 3000, Cost = 1,9021
Epoch 70, Update 3100, Cost = 1,5925
Epoch 72, Update 3200, Cost = 1,8163
Epoch 74, Update 3300, Cost = 1,7912
Epoch 77, Update 3400, Cost = 1,9172
Epoch 79, Update 3500, Cost = 1,4464
Epoch 81, Update 3600, Cost = 1,7621
Epoch 84, Update 3700, Cost = 1,8752
Epoch 86, Update 3800, Cost = 1,7666
Epoch 88, Update 3900, Cost = 1,6269
Epoch 90, Update 4000, Cost = 1,7417
Epoch 93, Update 4100, Cost = 1,9020
Epoch 95, Update 4200, Cost = 1,5925
Epoch 97, Update 4300, Cost = 1,8163
Epoch 99, Update 4400, Cost = 1,7912
Epoch 102, Update 4500, Cost = 1,9172
Epoch 104, Update 4600, Cost = 1,4464
Epoch 106, Update 4700, Cost = 1,7621
Epoch 109, Update 4800, Cost = 1,8752
Translations:
<s> Il cane è in giardino. </s>
<s> Il gatto è sul </s>
<s> Il gatto è è sedia. </s>
<s> Il gatto guarda l'uccello. </s>
<s> Dov'è il il </s>
<s> Chi un un gatto? </s>
             */


            /*
            Epoch 2, Update 100, Cost = 11,8332
Translations:
<s> Il cane cane è </s>
<s> Il gatto il </s>
<s> Lei il il </s>
<s> Il gatto il il </s>
<s> Lei il il </s>
<s> un un un </s>
Epoch 2, Update 100, Cost = 11,3291
Epoch 4, Update 200, Cost = 12,8480
Epoch 6, Update 300, Cost = 9,6390
Epoch 9, Update 400, Cost = 6,1589
Epoch 11, Update 500, Cost = 5,3310
Translations:
<s> Il cane è è in </s>
<s> Il gatto è sul sul </s>
<s> Il gatto gatto è sul </s>
<s> Il gatto guarda guarda l'uccello. </s>
<s> Dov'è dorme il </s>
<s> Lei ha un gatto? </s>
Epoch 2, Update 100, Cost = 12,4848
Epoch 4, Update 200, Cost = 13,6362
Epoch 6, Update 300, Cost = 11,6282
Epoch 9, Update 400, Cost = 5,9027
Epoch 11, Update 500, Cost = 5,2177
Epoch 13, Update 600, Cost = 4,5907
Epoch 15, Update 700, Cost = 3,9168
Epoch 18, Update 800, Cost = 2,3299
Translations:
<s> Il cane è in giardino. </s>
<s> Il gatto è sul </s>
<s> Il gatto sul sedia. </s>
<s> Il gatto guarda l'uccello. </s>
<s> È mangia gatto? </s>
<s> Lei un cane. </s>
Epoch 2, Update 100, Cost = 12,0825
Epoch 4, Update 200, Cost = 11,7361
Epoch 6, Update 300, Cost = 11,3464
Epoch 9, Update 400, Cost = 5,3572
Epoch 11, Update 500, Cost = 6,1524
Epoch 13, Update 600, Cost = 5,3356
Epoch 15, Update 700, Cost = 4,5446
Epoch 18, Update 800, Cost = 2,6563
Epoch 20, Update 900, Cost = 2,7862
Epoch 22, Update 1000, Cost = 2,5734
Epoch 24, Update 1100, Cost = 2,5558
Epoch 27, Update 1200, Cost = 1,7340
Epoch 29, Update 1300, Cost = 1,9541
Epoch 31, Update 1400, Cost = 2,2790
Epoch 34, Update 1500, Cost = 1,8840
Epoch 36, Update 1600, Cost = 2,0079
Epoch 38, Update 1700, Cost = 1,8819
Epoch 40, Update 1800, Cost = 2,1021
Epoch 43, Update 1900, Cost = 1,6276
Epoch 45, Update 2000, Cost = 1,8989
Epoch 47, Update 2100, Cost = 1,8952
Epoch 49, Update 2200, Cost = 2,0924
Epoch 52, Update 2300, Cost = 1,5124
Epoch 54, Update 2400, Cost = 1,7781
Translations:
<s> Il cane è in giardino. </s>
<s> Il gatto è sul tavolo. </s>
<s> Il gatto è al sedia. </s>
<s> Il gatto guarda l'uccello. </s>
<s> Chi un gatto? </s>
<s> Hai un cane. </s>
Epoch 2, Update 100, Cost = 11,9266
Epoch 4, Update 200, Cost = 12,6454
Epoch 6, Update 300, Cost = 10,2976
Epoch 9, Update 400, Cost = 6,0131
Epoch 11, Update 500, Cost = 5,5207
Epoch 13, Update 600, Cost = 3,8356
Epoch 15, Update 700, Cost = 3,3153
Epoch 18, Update 800, Cost = 2,2141
Epoch 20, Update 900, Cost = 1,9692
Epoch 22, Update 1000, Cost = 1,7755
Epoch 24, Update 1100, Cost = 1,8396
Epoch 27, Update 1200, Cost = 1,4535
Epoch 29, Update 1300, Cost = 1,2885
Epoch 31, Update 1400, Cost = 1,6078
Epoch 34, Update 1500, Cost = 1,3644
Epoch 36, Update 1600, Cost = 1,3024
Epoch 38, Update 1700, Cost = 1,2577
Epoch 40, Update 1800, Cost = 1,5430
Epoch 43, Update 1900, Cost = 1,2975
Epoch 45, Update 2000, Cost = 1,2842
Epoch 47, Update 2100, Cost = 1,3432
Epoch 49, Update 2200, Cost = 1,5139
Epoch 52, Update 2300, Cost = 1,2518
Epoch 54, Update 2400, Cost = 1,1677
Epoch 56, Update 2500, Cost = 1,5153
Epoch 59, Update 2600, Cost = 1,3063
Epoch 61, Update 2700, Cost = 1,2600
Epoch 63, Update 2800, Cost = 1,2306
Epoch 65, Update 2900, Cost = 1,5233
Translations:
<s> Il cane è in giardino. </s>
<s> Il gatto è sulla </s>
<s> Il suo è sulla sedia. </s>
<s> Il gatto guarda l'uccello. </s>
<s> Il ha gatto? gatto? </s>
<s> Hai un cane. </s>
Epoch 2, Update 100, Cost = 12,0794
Epoch 4, Update 200, Cost = 12,6638
Epoch 6, Update 300, Cost = 10,0005
Epoch 9, Update 400, Cost = 4,3036
Epoch 11, Update 500, Cost = 3,5129
Epoch 13, Update 600, Cost = 2,5434
Epoch 15, Update 700, Cost = 2,0174
Epoch 18, Update 800, Cost = 0,7447
Epoch 20, Update 900, Cost = 1,0536
Epoch 22, Update 1000, Cost = 0,8448
Epoch 24, Update 1100, Cost = 1,0273
Epoch 27, Update 1200, Cost = 0,5376
Epoch 29, Update 1300, Cost = 0,6549
Epoch 31, Update 1400, Cost = 0,7963
Epoch 34, Update 1500, Cost = 0,3481
Epoch 36, Update 1600, Cost = 0,5983
Epoch 38, Update 1700, Cost = 0,6250
Epoch 40, Update 1800, Cost = 0,7840
Epoch 43, Update 1900, Cost = 0,3135
Epoch 45, Update 2000, Cost = 0,6479
Epoch 47, Update 2100, Cost = 0,5918
Epoch 49, Update 2200, Cost = 0,8144
Epoch 52, Update 2300, Cost = 0,4541
Epoch 54, Update 2400, Cost = 0,5848
Epoch 56, Update 2500, Cost = 0,7358
Epoch 59, Update 2600, Cost = 0,3293
Epoch 61, Update 2700, Cost = 0,5768
Epoch 63, Update 2800, Cost = 0,6087
Epoch 65, Update 2900, Cost = 0,7705
Epoch 68, Update 3000, Cost = 0,3096
Epoch 70, Update 3100, Cost = 0,6429
Epoch 72, Update 3200, Cost = 0,5885
Epoch 74, Update 3300, Cost = 0,8115
Epoch 77, Update 3400, Cost = 0,4530
Epoch 79, Update 3500, Cost = 0,5839
Epoch 81, Update 3600, Cost = 0,7350
Epoch 84, Update 3700, Cost = 0,3290
Epoch 86, Update 3800, Cost = 0,5765
Translations:
<s> Il cane è in giardino. </s>
<s> Il gatto è sul tavolo. </s>
<s> Il gatto è sul letto. </s>
<s> Il gatto il accanto cane. </s>
<s> Chi ha gatto? gatto? </s>
<s> Hai un cane. </s>
Epoch 2, Update 100, Cost = 12,9325
Epoch 4, Update 200, Cost = 14,2181
Epoch 6, Update 300, Cost = 12,2676
Epoch 9, Update 400, Cost = 5,7616
Epoch 11, Update 500, Cost = 6,7102
Epoch 13, Update 600, Cost = 4,9890
Epoch 15, Update 700, Cost = 4,4595
Epoch 18, Update 800, Cost = 2,7099
Epoch 20, Update 900, Cost = 2,6181
Epoch 22, Update 1000, Cost = 2,4282
Epoch 24, Update 1100, Cost = 2,2581
Epoch 27, Update 1200, Cost = 1,7089
Epoch 29, Update 1300, Cost = 1,7278
Epoch 31, Update 1400, Cost = 1,8984
Epoch 34, Update 1500, Cost = 1,4003
Epoch 36, Update 1600, Cost = 1,6476
Epoch 38, Update 1700, Cost = 1,6569
Epoch 40, Update 1800, Cost = 1,8268
Epoch 43, Update 1900, Cost = 1,3732
Epoch 45, Update 2000, Cost = 1,6438
Epoch 47, Update 2100, Cost = 1,6916
Epoch 49, Update 2200, Cost = 1,7869
Epoch 52, Update 2300, Cost = 1,4241
Epoch 54, Update 2400, Cost = 1,5414
Epoch 56, Update 2500, Cost = 1,7454
Epoch 59, Update 2600, Cost = 1,2990
Epoch 61, Update 2700, Cost = 1,5814
Epoch 63, Update 2800, Cost = 1,6138
Epoch 65, Update 2900, Cost = 1,7928
Epoch 68, Update 3000, Cost = 1,3521
Epoch 70, Update 3100, Cost = 1,6296
Epoch 72, Update 3200, Cost = 1,6822
Epoch 74, Update 3300, Cost = 1,7797
Epoch 77, Update 3400, Cost = 1,4198
Epoch 79, Update 3500, Cost = 1,5386
Epoch 81, Update 3600, Cost = 1,7435
Epoch 84, Update 3700, Cost = 1,2979
Epoch 86, Update 3800, Cost = 1,5806
Epoch 88, Update 3900, Cost = 1,6134
Epoch 90, Update 4000, Cost = 1,7925
Epoch 93, Update 4100, Cost = 1,3520
Epoch 95, Update 4200, Cost = 1,6295
Epoch 97, Update 4300, Cost = 1,6822
Epoch 99, Update 4400, Cost = 1,7797
Epoch 102, Update 4500, Cost = 1,4198
Epoch 104, Update 4600, Cost = 1,5386
Epoch 106, Update 4700, Cost = 1,7435
Epoch 109, Update 4800, Cost = 1,2979
Translations:
<s> Il cane è in giardino. </s>
<s> Il gatto è sul </s>
<s> Il gatto è sul </s>
<s> Il gatto guarda l'uccello. </s>
<s> Il cane è </s>
<s> Chi un un </s> 
             */


            Console.ReadLine();
        }
    }
}
