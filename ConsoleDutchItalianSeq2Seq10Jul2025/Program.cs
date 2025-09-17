using Seq2SeqSharp;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Enums;
using Seq2SeqSharp.LearningRate;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;

// ChatGPT, modified
namespace ConsoleDutchItalianSeq2Seq10Jul2025
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

                var trainData = new List<(string src, string tgt)>
                {
                    ("Slaap lekker", "Dormi bene"),
                    ("Eet smakelijk", "Buon appetito"),
                    ("Welkom", "Benvenuto"),
                    ("Dank je wel", "Grazie"),
                    ("Alsjeblieft", "Prego"),
                    ("Sorry", "Mi dispiace"),
                    ("Geen probleem", "Nessun problema"),
                    ("Ik begrijp het", "Lo capisco"),
                    ("Hoe oud ben je ?", "Quanti anni hai ?"),
                    ("Ik ben twintig jaar oud", "Ho venti anni"),
                    ("Ik studeer aan de universiteit", "Studio all'università"),
                    ("Wat is dit ?", "Che cos'è questo ?"),
                    ("Dat is mooi", "È bello"),
                    ("Ik hou van muziek", "Amo la musica"),
                    ("Ik speel gitaar", "Suono la chitarra"),
                    ("Hij speelt voetbal", "Lui gioca a calcio"),
                    ("We gaan naar het park", "Andiamo al parco"),
                    ("Ik wil naar huis", "Voglio andare a casa"),
                    ("De kat slaapt", "Il gatto dorme"),
                    ("De hond blaft", "Il cane abbaia"),
                    ("Ik lees een boek", "Sto leggendo un libro"),
                    ("We kijken een film", "Guardiamo un film"),
                    ("Wat wil je doen ?", "Cosa vuoi fare ?"),
                    ("Laten we gaan wandelen", "Facciamo una passeggiata"),
                    ("Ik ben aan het koken", "Sto cucinando"),
                    ("Het ruikt lekker", "Ha un buon profumo"),
                    ("Ben je klaar ?", "Sei pronto ?"),
                    ("Ik weet het niet", "Non lo so"),
                    ("Ik denk van wel", "Penso di sì"),
                    ("Misschien", "Forse"),
                    ("Ik ben bang", "Ho paura"),
                    ("Wees voorzichtig", "Stai attento"),
                    ("Dat is gevaarlijk", "È pericoloso"),
                    ("Ik hou van reizen", "Mi piace viaggiare"),
                    ("Ik ga met de trein", "Prendo il treno"),
                    ("Waar gaan we heen ?", "Dove andiamo ?"),
                    ("Wat is jouw favoriete kleur ?", "Qual è il tuo colore preferito ?"),
                    ("Mijn favoriete kleur is blauw", "Il mio colore preferito è il blu"),
                    ("Ik luister naar de radio", "Ascolto la radio"),
                    ("De kinderen spelen buiten", "I bambini giocano fuori")
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
                    HiddenSize = 512,
                    SrcEmbeddingDim = 512,
                    TgtEmbeddingDim = 512,
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
                    "Ik weet het niet",
                    "Wat is dit ?",
                    "Ik luister naar de radio",
                    "Ik begrijp het",
                    "Hij speelt gitaar"
                 });

                inferModel.Test(
                    inputTestFile: testInputPath,
                    outputFile: testOutputPath,
                    batchSize: 1,
                    decodingOptions: opts.CreateDecodingOptions(),
                    srcSpmPath: null,
                    tgtSpmPath: null); // We are not using SentencePiece

                Console.WriteLine("\nTranslations:");
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
     Translations:
<s> Sto </s>
<s> Il ? ? </s>
<s> Il </s>
<s> Il </s>
<s> Il </s>
Epoch 5, Update 100, Cost = 2,6172
Epoch 10, Update 200, Cost = 1,2929
Translations:
<s> Non lo so </s>
<s> Dove andiamo ? ? </s>
<s> Ascolto la radio </s>
<s> Sto un </s>
<s> Suono la </s>
Epoch 5, Update 100, Cost = 7,1080
Epoch 10, Update 200, Cost = 3,1120
Epoch 15, Update 300, Cost = 0,7558
Translations:
<s> Non lo so </s>
<s> Qual è è ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ?
<s> Ascolto la radio </s>
<s> Dove bambini </s>
<s> I bambini colore fuori </s>
Epoch 5, Update 100, Cost = 7,3399
Epoch 10, Update 200, Cost = 3,3732
Epoch 15, Update 300, Cost = 0,6772
Epoch 21, Update 400, Cost = 0,0959
Epoch 26, Update 500, Cost = 1,7840
Epoch 31, Update 600, Cost = 0,5269
Epoch 36, Update 700, Cost = 0,2943
Epoch 42, Update 800, Cost = 0,2957
Epoch 47, Update 900, Cost = 0,2101
Epoch 52, Update 1000, Cost = 0,1780
Translations:
<s> Non lo so </s>
<s> Che cos'è questo ? </s>
<s> Ascolto la radio </s>
<s> Lo capisco </s>
<s> Suono la chitarra </s>
Epoch 5, Update 100, Cost = 12,0068
Epoch 10, Update 200, Cost = 1,2938
Epoch 15, Update 300, Cost = 0,3785
Epoch 21, Update 400, Cost = 0,0991
Epoch 26, Update 500, Cost = 1,9359
Epoch 31, Update 600, Cost = 0,6056
Epoch 36, Update 700, Cost = 0,3158
Epoch 42, Update 800, Cost = 0,2495
Epoch 47, Update 900, Cost = 0,2294
Epoch 52, Update 1000, Cost = 0,1750
Epoch 57, Update 1100, Cost = 0,1735
Epoch 63, Update 1200, Cost = 0,1417
Translations:
<s> Non lo so </s>
<s> Che cos'è questo ? </s>
<s> Ascolto la radio </s>
<s> Lo capisco </s>
<s> Lui gioca a calcio </s>
Epoch 5, Update 100, Cost = 8,4198
Epoch 10, Update 200, Cost = 5,4480
Epoch 15, Update 300, Cost = 10,4059
Epoch 21, Update 400, Cost = 1,2846
Epoch 26, Update 500, Cost = 0,5156
Epoch 31, Update 600, Cost = 0,3684
Epoch 36, Update 700, Cost = 2,2438
Epoch 42, Update 800, Cost = 2,6051
Epoch 47, Update 900, Cost = 1,1158
Epoch 52, Update 1000, Cost = 0,9638
Epoch 57, Update 1100, Cost = 0,6363
Epoch 63, Update 1200, Cost = 0,7253
Epoch 68, Update 1300, Cost = 0,5425
Epoch 73, Update 1400, Cost = 0,4278
Epoch 78, Update 1500, Cost = 0,4861
Epoch 84, Update 1600, Cost = 0,4413
Translations:
<s> Non lo so </s>
<s> Che cos'è questo ? </s>
<s> Ascolto la radio </s>
<s> Lo capisco </s>
<s> Lui gioca a </s>
Epoch 5, Update 100, Cost = 4,8418
Epoch 10, Update 200, Cost = 1,9517
Epoch 15, Update 300, Cost = 0,3079
Epoch 21, Update 400, Cost = 9,2828
Epoch 26, Update 500, Cost = 1,4269
Epoch 31, Update 600, Cost = 0,5057
Epoch 36, Update 700, Cost = 0,2767
Epoch 42, Update 800, Cost = 0,1608
Epoch 47, Update 900, Cost = 0,1828
Epoch 52, Update 1000, Cost = 0,1596
Epoch 57, Update 1100, Cost = 0,2073
Epoch 63, Update 1200, Cost = 0,1298
Epoch 68, Update 1300, Cost = 0,1222
Epoch 73, Update 1400, Cost = 0,1182
Epoch 78, Update 1500, Cost = 0,2209
Epoch 84, Update 1600, Cost = 0,1004
Epoch 89, Update 1700, Cost = 0,1161
Epoch 94, Update 1800, Cost = 0,1065
Epoch 99, Update 1900, Cost = 0,2015
Epoch 105, Update 2000, Cost = 0,1040
Translations:
<s> Non lo so </s>
<s> Che cos'è questo ? </s>
<s> Ascolto la radio </s>
<s> Lo capisco </s>
<s> Lui gioca a </s>
Epoch 5, Update 100, Cost = 2,6990
Epoch 10, Update 200, Cost = 4,4532
Epoch 15, Update 300, Cost = 1,8167
Epoch 21, Update 400, Cost = 0,1505
Epoch 26, Update 500, Cost = 5,5037
Epoch 31, Update 600, Cost = 1,9102
Epoch 36, Update 700, Cost = 0,8940
Epoch 42, Update 800, Cost = 0,9912
Epoch 47, Update 900, Cost = 0,7957
Epoch 52, Update 1000, Cost = 0,5126
Epoch 57, Update 1100, Cost = 0,4777
Epoch 63, Update 1200, Cost = 0,5000
Epoch 68, Update 1300, Cost = 0,3649
Epoch 73, Update 1400, Cost = 0,3161
Epoch 78, Update 1500, Cost = 0,3762
Epoch 84, Update 1600, Cost = 0,3243
Epoch 89, Update 1700, Cost = 0,3238
Epoch 94, Update 1800, Cost = 0,2894
Epoch 99, Update 1900, Cost = 0,3493
Epoch 105, Update 2000, Cost = 0,3219
Epoch 110, Update 2100, Cost = 0,3120
Epoch 115, Update 2200, Cost = 0,2754
Epoch 121, Update 2300, Cost = 0,4148
Epoch 126, Update 2400, Cost = 0,3105
Epoch 131, Update 2500, Cost = 0,2952
Epoch 136, Update 2600, Cost = 0,2664
Epoch 142, Update 2700, Cost = 0,3814
Epoch 147, Update 2800, Cost = 0,3127
Epoch 152, Update 2900, Cost = 0,2922
Epoch 157, Update 3000, Cost = 0,3235
Epoch 163, Update 3100, Cost = 0,3707
Epoch 168, Update 3200, Cost = 0,3115
Epoch 173, Update 3300, Cost = 0,2845
Epoch 178, Update 3400, Cost = 0,3488
Epoch 184, Update 3500, Cost = 0,3067
Epoch 189, Update 3600, Cost = 0,3129
Epoch 194, Update 3700, Cost = 0,2827
Epoch 199, Update 3800, Cost = 0,3437
Translations:
<s> Non lo so </s>
<s> Che cos'è questo ? </s>
<s> Ascolto la radio </s>
<s> Lo capisco </s>
<s> Suono la chitarra </s>
Epoch 5, Update 100, Cost = 6,5803
Epoch 10, Update 200, Cost = 1,6152
Epoch 15, Update 300, Cost = 14,2041
Epoch 21, Update 400, Cost = 1,5231
Epoch 26, Update 500, Cost = 0,5271
Epoch 31, Update 600, Cost = 0,3140
Epoch 36, Update 700, Cost = 1,4771
Epoch 42, Update 800, Cost = 1,1632
Epoch 47, Update 900, Cost = 0,5777
Epoch 52, Update 1000, Cost = 0,4010
Epoch 57, Update 1100, Cost = 0,3820
Epoch 63, Update 1200, Cost = 0,3275
Epoch 68, Update 1300, Cost = 0,2674
Epoch 73, Update 1400, Cost = 0,2375
Epoch 78, Update 1500, Cost = 0,3000
Epoch 84, Update 1600, Cost = 0,2179
Epoch 89, Update 1700, Cost = 0,2274
Epoch 94, Update 1800, Cost = 0,2098
Epoch 99, Update 1900, Cost = 0,2726
Epoch 105, Update 2000, Cost = 0,2090
Epoch 110, Update 2100, Cost = 0,2182
Epoch 115, Update 2200, Cost = 0,2019
Epoch 121, Update 2300, Cost = 0,2407
Epoch 126, Update 2400, Cost = 0,2061
Epoch 131, Update 2500, Cost = 0,2093
Epoch 136, Update 2600, Cost = 0,1954
Epoch 142, Update 2700, Cost = 0,2085
Epoch 147, Update 2800, Cost = 0,2108
Epoch 152, Update 2900, Cost = 0,2112
Epoch 157, Update 3000, Cost = 0,2512
Epoch 163, Update 3100, Cost = 0,2335
Epoch 168, Update 3200, Cost = 0,2140
Epoch 173, Update 3300, Cost = 0,2078
Epoch 178, Update 3400, Cost = 0,2742
Epoch 184, Update 3500, Cost = 0,2031
Epoch 189, Update 3600, Cost = 0,2174
Epoch 194, Update 3700, Cost = 0,2039
Epoch 199, Update 3800, Cost = 0,2674
Epoch 205, Update 3900, Cost = 0,2059
Epoch 210, Update 4000, Cost = 0,2163
Epoch 215, Update 4100, Cost = 0,2007
Epoch 221, Update 4200, Cost = 0,2395
Epoch 226, Update 4300, Cost = 0,2055
Epoch 231, Update 4400, Cost = 0,2090
Epoch 236, Update 4500, Cost = 0,1952
Epoch 242, Update 4600, Cost = 0,2083
Epoch 247, Update 4700, Cost = 0,2107
Epoch 252, Update 4800, Cost = 0,2112
Translations:
<s> Non lo so </s>
<s> Che cos'è questo ? </s>
<s> Ascolto la radio </s>
<s> Lo capisco </s>
<s> Lui gioca a calcio </s>
Epoch 5, Update 100, Cost = 2,3416
Epoch 10, Update 200, Cost = 3,0892
Epoch 15, Update 300, Cost = 0,7815
Epoch 21, Update 400, Cost = 0,1218
Epoch 26, Update 500, Cost = 6,0332
Epoch 31, Update 600, Cost = 0,7364
Epoch 36, Update 700, Cost = 0,3401
Epoch 42, Update 800, Cost = 0,1993
Epoch 47, Update 900, Cost = 0,1849
Epoch 52, Update 1000, Cost = 0,1509
Epoch 57, Update 1100, Cost = 0,2021
Epoch 63, Update 1200, Cost = 0,0918
Epoch 68, Update 1300, Cost = 0,1120
Epoch 73, Update 1400, Cost = 0,1029
Epoch 78, Update 1500, Cost = 0,1830
Epoch 84, Update 1600, Cost = 0,0741
Epoch 89, Update 1700, Cost = 0,1011
Epoch 94, Update 1800, Cost = 0,0913
Epoch 99, Update 1900, Cost = 0,1661
Epoch 105, Update 2000, Cost = 0,0860
Epoch 110, Update 2100, Cost = 0,0975
Epoch 115, Update 2200, Cost = 0,0891
Epoch 121, Update 2300, Cost = 0,0883
Epoch 126, Update 2400, Cost = 0,0876
Epoch 131, Update 2500, Cost = 0,0930
Epoch 136, Update 2600, Cost = 0,0864
Epoch 142, Update 2700, Cost = 0,0639
Epoch 147, Update 2800, Cost = 0,0918
Epoch 152, Update 2900, Cost = 0,0949
Epoch 157, Update 3000, Cost = 0,1400
Epoch 163, Update 3100, Cost = 0,0739
Epoch 168, Update 3200, Cost = 0,0959
Epoch 173, Update 3300, Cost = 0,0928
Epoch 178, Update 3400, Cost = 0,1672
Epoch 184, Update 3500, Cost = 0,0706
Epoch 189, Update 3600, Cost = 0,0978
Epoch 194, Update 3700, Cost = 0,0893
Epoch 199, Update 3800, Cost = 0,1629
Epoch 205, Update 3900, Cost = 0,0850
Epoch 210, Update 4000, Cost = 0,0969
Epoch 215, Update 4100, Cost = 0,0887
Epoch 221, Update 4200, Cost = 0,0880
Epoch 226, Update 4300, Cost = 0,0874
Epoch 231, Update 4400, Cost = 0,0928
Epoch 236, Update 4500, Cost = 0,0864
Epoch 242, Update 4600, Cost = 0,0639
Epoch 247, Update 4700, Cost = 0,0918
Epoch 252, Update 4800, Cost = 0,0949
Epoch 257, Update 4900, Cost = 0,1400
Epoch 263, Update 5000, Cost = 0,0739
Epoch 268, Update 5100, Cost = 0,0959
Epoch 273, Update 5200, Cost = 0,0928
Epoch 278, Update 5300, Cost = 0,1672
Epoch 284, Update 5400, Cost = 0,0706
Epoch 289, Update 5500, Cost = 0,0978
Epoch 294, Update 5600, Cost = 0,0893
Epoch 299, Update 5700, Cost = 0,1629
Epoch 305, Update 5800, Cost = 0,0850
Epoch 310, Update 5900, Cost = 0,0969
Epoch 315, Update 6000, Cost = 0,0887
Epoch 321, Update 6100, Cost = 0,0880
Epoch 326, Update 6200, Cost = 0,0874
Epoch 331, Update 6300, Cost = 0,0928
Epoch 336, Update 6400, Cost = 0,0864
Epoch 342, Update 6500, Cost = 0,0639
Epoch 347, Update 6600, Cost = 0,0918
Epoch 352, Update 6700, Cost = 0,0949
Epoch 357, Update 6800, Cost = 0,1400
Epoch 363, Update 6900, Cost = 0,0739
Translations:
<s> Non lo so </s>
<s> Che cos'è questo ? </s>
<s> Ascolto la radio </s>
<s> Lo capisco </s>
<s> Lui gioca a calcio </s>
     */


            /*
            Translations:
<s> Sto lo </s>
<s> Cosa è ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ?
<s> Ascolto la </s>
<s> Sto </s>
<s> Sto il </s>
Epoch 5, Update 100, Cost = 4,5432
Epoch 10, Update 200, Cost = 3,5089

Translations:
<s> Non lo so </s>
<s> Qual è il colore colore </s>
<s> Ascolto la radio </s>
<s> Non lo </s>
<s> Ho gioca </s>
Epoch 5, Update 100, Cost = 4,4064
Epoch 10, Update 200, Cost = 5,6846
Epoch 15, Update 300, Cost = 0,4385

Translations:
<s> Non lo so </s>
<s> Qual è il tuo ? ? </s>
<s> Ascolto la radio </s>
<s> Non lo </s>
<s> Lui gioca a </s>
Epoch 5, Update 100, Cost = 5,5643
Epoch 10, Update 200, Cost = 5,9815
Epoch 15, Update 300, Cost = 6,8164
Epoch 21, Update 400, Cost = 0,5765
Epoch 26, Update 500, Cost = 0,3988
Epoch 31, Update 600, Cost = 0,2831
Epoch 36, Update 700, Cost = 0,1864
Epoch 42, Update 800, Cost = 0,0722
Epoch 47, Update 900, Cost = 0,1105
Epoch 52, Update 1000, Cost = 0,1093

Translations:
<s> Non lo so </s>
<s> Cosa vuoi ? ? ? ? ? ? ? ? ? </s>
<s> Ascolto la radio </s>
<s> Non lo </s>
<s> Facciamo il radio </s>
Epoch 5, Update 100, Cost = 4,8141
Epoch 10, Update 200, Cost = 3,6454
Epoch 15, Update 300, Cost = 2,5173
Epoch 21, Update 400, Cost = 0,1375
Epoch 26, Update 500, Cost = 0,1202
Epoch 31, Update 600, Cost = 0,1043
Epoch 36, Update 700, Cost = 0,0814
Epoch 42, Update 800, Cost = 0,0371
Epoch 47, Update 900, Cost = 0,0492
Epoch 52, Update 1000, Cost = 0,0497
Epoch 57, Update 1100, Cost = 5,3956
Epoch 63, Update 1200, Cost = 6,7984

Translations:
<s> Non lo so </s>
<s> Qual è il ? </s>
<s> Ascolto la radio </s>
<s> Non lo </s>
<s> Lui gioca a </s>
Epoch 5, Update 100, Cost = 2,3278
Epoch 10, Update 200, Cost = 1,0495
Epoch 15, Update 300, Cost = 0,2394
Epoch 21, Update 400, Cost = 0,0622
Epoch 26, Update 500, Cost = 0,0692
Epoch 31, Update 600, Cost = 0,0592
Epoch 36, Update 700, Cost = 0,0467
Epoch 42, Update 800, Cost = 9,7300
Epoch 47, Update 900, Cost = 7,6297
Epoch 52, Update 1000, Cost = 6,5198
Epoch 57, Update 1100, Cost = 5,9563
Epoch 63, Update 1200, Cost = 5,3092
Epoch 68, Update 1300, Cost = 5,0148
Epoch 73, Update 1400, Cost = 4,7954
Epoch 78, Update 1500, Cost = 5,0734
Epoch 84, Update 1600, Cost = 4,1260

Translations:
<s> Non lo so </s>
<s> Che cos'è ? </s>
<s> Il la radio </s>
<s> Il la </s>
<s> Il gatto </s>
Epoch 5, Update 100, Cost = 3,8388
Epoch 10, Update 200, Cost = 2,0070
Epoch 15, Update 300, Cost = 0,3861
Epoch 21, Update 400, Cost = 0,0737
Epoch 26, Update 500, Cost = 0,0886
Epoch 31, Update 600, Cost = 0,0718
Epoch 36, Update 700, Cost = 0,0557
Epoch 42, Update 800, Cost = 0,0259
Epoch 47, Update 900, Cost = 12,7516
Epoch 52, Update 1000, Cost = 11,4065
Epoch 57, Update 1100, Cost = 10,9465
Epoch 63, Update 1200, Cost = 7,9965
Epoch 68, Update 1300, Cost = 8,7136
Epoch 73, Update 1400, Cost = 8,8406
Epoch 78, Update 1500, Cost = 10,0424
Epoch 84, Update 1600, Cost = 6,6560
Epoch 89, Update 1700, Cost = 8,1600
Epoch 94, Update 1800, Cost = 9,4722
Epoch 99, Update 1900, Cost = 10,7994
Epoch 105, Update 2000, Cost = 7,5476

Translations:
<s> Nessun la </s>
<s> Prego questo </s>
<s> Ascolto la radio </s>
<s> Nessun la </s>
<s> Quanti </s>
Epoch 5, Update 100, Cost = 4,0374
Epoch 10, Update 200, Cost = 1,9130
Epoch 15, Update 300, Cost = 0,5444
Epoch 21, Update 400, Cost = 0,1235
Epoch 26, Update 500, Cost = 0,1015
Epoch 31, Update 600, Cost = 16,0637
Epoch 36, Update 700, Cost = 8,6725
Epoch 42, Update 800, Cost = 5,2793
Epoch 47, Update 900, Cost = 3,8718
Epoch 52, Update 1000, Cost = 3,5230
Epoch 57, Update 1100, Cost = 3,1960
Epoch 63, Update 1200, Cost = 2,4172
Epoch 68, Update 1300, Cost = 2,3247
Epoch 73, Update 1400, Cost = 2,2762
Epoch 78, Update 1500, Cost = 2,4939
Epoch 84, Update 1600, Cost = 4,8759
Epoch 89, Update 1700, Cost = 6,5578
Epoch 94, Update 1800, Cost = 5,4479
Epoch 99, Update 1900, Cost = 4,9699
Epoch 105, Update 2000, Cost = 5,3860
Epoch 110, Update 2100, Cost = 5,7385
Epoch 115, Update 2200, Cost = 4,9715
Epoch 121, Update 2300, Cost = 3,9299
Epoch 126, Update 2400, Cost = 5,2061
Epoch 131, Update 2500, Cost = 5,2707
Epoch 136, Update 2600, Cost = 4,4725
Epoch 142, Update 2700, Cost = 3,7615
Epoch 147, Update 2800, Cost = 5,1593
Epoch 152, Update 2900, Cost = 5,1807
Epoch 157, Update 3000, Cost = 4,5514
Epoch 163, Update 3100, Cost = 4,5682
Epoch 168, Update 3200, Cost = 5,3071
Epoch 173, Update 3300, Cost = 4,9245
Epoch 178, Update 3400, Cost = 4,5971
Epoch 184, Update 3500, Cost = 4,3810
Epoch 189, Update 3600, Cost = 5,4847
Epoch 194, Update 3700, Cost = 4,7760
Epoch 199, Update 3800, Cost = 4,4449

Translations:
<s> Non lo so </s>
<s> Che cos'è questo questo ? ? ? ? ? </s>
<s> Ascolto la radio </s>
<s> Lo capisco </s>
<s> Lui gioca a calcio calcio </s>
             */


            /*
             Translations:
<s> Non </s>
<s> è è è ? ? </s>
<s> Ascolto </s>
<s> </s>
<s> Il </s>
Epoch 5, Update 100, Cost = 4,3020
Epoch 10, Update 200, Cost = 2,4191

Translations:
<s> Non lo so </s>
<s> Qual è il ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ?
<s> Ascolto la radio </s>
<s> Prendo paura </s>
<s> Ascolto paura </s>
Epoch 5, Update 100, Cost = 5,2539
Epoch 10, Update 200, Cost = 2,3852
Epoch 15, Update 300, Cost = 0,4958

Translations:
<s> Non lo so </s>
<s> Qual è il tuo tuo ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ?
<s> Ascolto la radio </s>
<s> Non lo </s>
<s> Prendo andiamo ? </s>
Epoch 5, Update 100, Cost = 2,6093
Epoch 10, Update 200, Cost = 1,0310
Epoch 15, Update 300, Cost = 8,1704
Epoch 21, Update 400, Cost = 0,7396
Epoch 26, Update 500, Cost = 0,3406
Epoch 31, Update 600, Cost = 0,2690
Epoch 36, Update 700, Cost = 0,1809
Epoch 42, Update 800, Cost = 0,0783
Epoch 47, Update 900, Cost = 0,1027
Epoch 52, Update 1000, Cost = 0,1039

Translations:
<s> Non lo so </s>
<s> Qual è è tuo colore colore colore ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? è è è è è è colore colore ? ? ? ? ? ? ? ? ? ? ? ? colore è è è è è colore colore ? ? ? ? ? ? ? ?
<s> Ascolto la radio </s>
<s> Lo </s>
<s> Lui gioca a calcio calcio </s>
Epoch 5, Update 100, Cost = 3,8221
Epoch 10, Update 200, Cost = 6,1657
Epoch 15, Update 300, Cost = 1,6997
Epoch 21, Update 400, Cost = 0,1609
Epoch 26, Update 500, Cost = 0,1567
Epoch 31, Update 600, Cost = 0,1231
Epoch 36, Update 700, Cost = 0,0968
Epoch 42, Update 800, Cost = 0,0416
Epoch 47, Update 900, Cost = 0,0569
Epoch 52, Update 1000, Cost = 0,0594
Epoch 57, Update 1100, Cost = 5,0222
Epoch 63, Update 1200, Cost = 4,8968

Translations:
<s> Non lo so </s>
<s> Qual è il ? ? </s>
<s> Ascolto la radio </s>
<s> Sto cucinando </s>
<s> Lui gioca a </s>
Epoch 5, Update 100, Cost = 3,9189
Epoch 10, Update 200, Cost = 0,9519
Epoch 15, Update 300, Cost = 0,2464
Epoch 21, Update 400, Cost = 0,0675
Epoch 26, Update 500, Cost = 0,0731
Epoch 31, Update 600, Cost = 0,0643
Epoch 36, Update 700, Cost = 0,0496
Epoch 42, Update 800, Cost = 9,8160
Epoch 47, Update 900, Cost = 8,3580
Epoch 52, Update 1000, Cost = 7,7631
Epoch 57, Update 1100, Cost = 7,2003
Epoch 63, Update 1200, Cost = 5,5395
Epoch 68, Update 1300, Cost = 5,8681
Epoch 73, Update 1400, Cost = 5,7012
Epoch 78, Update 1500, Cost = 5,8387
Epoch 84, Update 1600, Cost = 4,6770

Translations:
<s> Ho lo so </s>
<s> Che questo ? </s>
<s> Ho la radio </s>
<s> Benvenuto la </s>
<s> Andiamo al </s>
Epoch 5, Update 100, Cost = 3,7616
Epoch 10, Update 200, Cost = 1,5585
Epoch 15, Update 300, Cost = 0,3211
Epoch 21, Update 400, Cost = 0,0816
Epoch 26, Update 500, Cost = 0,0825
Epoch 31, Update 600, Cost = 0,0771
Epoch 36, Update 700, Cost = 0,0622
Epoch 42, Update 800, Cost = 0,0309
Epoch 47, Update 900, Cost = 0,0444
Epoch 52, Update 1000, Cost = 2,9374
Epoch 57, Update 1100, Cost = 1,7346
Epoch 63, Update 1200, Cost = 4,0297
Epoch 68, Update 1300, Cost = 2,0787
Epoch 73, Update 1400, Cost = 1,2636
Epoch 78, Update 1500, Cost = 1,0154
Epoch 84, Update 1600, Cost = 2,3330
Epoch 89, Update 1700, Cost = 1,4818
Epoch 94, Update 1800, Cost = 0,9958
Epoch 99, Update 1900, Cost = 0,7916
Epoch 105, Update 2000, Cost = 1,9212

Translations:
<s> Non lo so </s>
<s> Qual cos'è questo ? </s>
<s> Ascolto la radio </s>
<s> Lo capisco </s>
<s> Lui la chitarra </s>
Epoch 5, Update 100, Cost = 3,7135
Epoch 10, Update 200, Cost = 1,3249
Epoch 15, Update 300, Cost = 0,3712
Epoch 21, Update 400, Cost = 0,0850
Epoch 26, Update 500, Cost = 0,1094
Epoch 31, Update 600, Cost = 0,0854
Epoch 36, Update 700, Cost = 0,0662
Epoch 42, Update 800, Cost = 10,1141
Epoch 47, Update 900, Cost = 8,3207
Epoch 52, Update 1000, Cost = 7,2500
Epoch 57, Update 1100, Cost = 6,6997
Epoch 63, Update 1200, Cost = 5,9859
Epoch 68, Update 1300, Cost = 5,6860
Epoch 73, Update 1400, Cost = 5,3441
Epoch 78, Update 1500, Cost = 5,8044
Epoch 84, Update 1600, Cost = 4,8977
Epoch 89, Update 1700, Cost = 8,2960
Epoch 94, Update 1800, Cost = 7,1371
Epoch 99, Update 1900, Cost = 7,3032
Epoch 105, Update 2000, Cost = 6,6374
Epoch 110, Update 2100, Cost = 7,5641
Epoch 115, Update 2200, Cost = 6,7305
Epoch 121, Update 2300, Cost = 5,7430
Epoch 126, Update 2400, Cost = 6,9102
Epoch 131, Update 2500, Cost = 7,1342
Epoch 136, Update 2600, Cost = 6,3990
Epoch 142, Update 2700, Cost = 5,4617
Epoch 147, Update 2800, Cost = 7,5364
Epoch 152, Update 2900, Cost = 7,0090
Epoch 157, Update 3000, Cost = 6,7781
Epoch 163, Update 3100, Cost = 6,1197
Epoch 168, Update 3200, Cost = 7,8405
Epoch 173, Update 3300, Cost = 6,9033
Epoch 178, Update 3400, Cost = 7,1585
Epoch 184, Update 3500, Cost = 5,8676
Epoch 189, Update 3600, Cost = 7,6311
Epoch 194, Update 3700, Cost = 6,7152
Epoch 199, Update 3800, Cost = 6,9663

Translations:
<s> Non lo so </s>
<s> Che cos'è ? </s>
<s> Ascolto la radio </s>
<s> Sto la </s>
<s> Sto leggendo a </s>
Epoch 5, Update 100, Cost = 1,5696
Epoch 10, Update 200, Cost = 0,8621
Epoch 15, Update 300, Cost = 0,1684
Epoch 21, Update 400, Cost = 0,0429
Epoch 26, Update 500, Cost = 0,0546
Epoch 31, Update 600, Cost = 15,8584
Epoch 36, Update 700, Cost = 7,4488
Epoch 42, Update 800, Cost = 4,5741
Epoch 47, Update 900, Cost = 3,3166
Epoch 52, Update 1000, Cost = 2,8079
Epoch 57, Update 1100, Cost = 2,4693
Epoch 63, Update 1200, Cost = 2,0895
Epoch 68, Update 1300, Cost = 1,9541
Epoch 73, Update 1400, Cost = 1,7842
Epoch 78, Update 1500, Cost = 1,8803
Epoch 84, Update 1600, Cost = 3,7448
Epoch 89, Update 1700, Cost = 3,4844
Epoch 94, Update 1800, Cost = 2,8633
Epoch 99, Update 1900, Cost = 2,8808
Epoch 105, Update 2000, Cost = 3,4217
Epoch 110, Update 2100, Cost = 3,1864
Epoch 115, Update 2200, Cost = 2,6485
Epoch 121, Update 2300, Cost = 3,5475
Epoch 126, Update 2400, Cost = 3,1857
Epoch 131, Update 2500, Cost = 2,8613
Epoch 136, Update 2600, Cost = 2,4492
Epoch 142, Update 2700, Cost = 3,3012
Epoch 147, Update 2800, Cost = 3,0540
Epoch 152, Update 2900, Cost = 2,8335
Epoch 157, Update 3000, Cost = 2,7447
Epoch 163, Update 3100, Cost = 3,4163
Epoch 168, Update 3200, Cost = 3,2079
Epoch 173, Update 3300, Cost = 2,7386
Epoch 178, Update 3400, Cost = 2,7591
Epoch 184, Update 3500, Cost = 3,2993
Epoch 189, Update 3600, Cost = 3,1373
Epoch 194, Update 3700, Cost = 2,6559
Epoch 199, Update 3800, Cost = 2,6868
Epoch 205, Update 3900, Cost = 3,2714
Epoch 210, Update 4000, Cost = 3,0432
Epoch 215, Update 4100, Cost = 2,5563
Epoch 221, Update 4200, Cost = 3,4827
Epoch 226, Update 4300, Cost = 3,0959
Epoch 231, Update 4400, Cost = 2,8510
Epoch 236, Update 4500, Cost = 2,4436
Epoch 242, Update 4600, Cost = 3,2977
Epoch 247, Update 4700, Cost = 3,0501
Epoch 252, Update 4800, Cost = 2,8306

Translations:
<s> Non lo so </s>
<s> Che cos'è questo ? </s>
<s> Ascolto la radio </s>
<s> Lo capisco </s>
<s> Lui gioca a calcio </s>
Epoch 5, Update 100, Cost = 3,5375
Epoch 10, Update 200, Cost = 2,8077
Epoch 15, Update 300, Cost = 0,2534
Epoch 21, Update 400, Cost = 12,0495
Epoch 26, Update 500, Cost = 3,4230
Epoch 31, Update 600, Cost = 1,4630
Epoch 36, Update 700, Cost = 0,8744
Epoch 42, Update 800, Cost = 0,3773
Epoch 47, Update 900, Cost = 0,3366
Epoch 52, Update 1000, Cost = 0,2978
Epoch 57, Update 1100, Cost = 0,3263
Epoch 63, Update 1200, Cost = 0,1805
Epoch 68, Update 1300, Cost = 0,1984
Epoch 73, Update 1400, Cost = 3,3664
Epoch 78, Update 1500, Cost = 2,4137
Epoch 84, Update 1600, Cost = 3,4199
Epoch 89, Update 1700, Cost = 3,0497
Epoch 94, Update 1800, Cost = 2,3311
Epoch 99, Update 1900, Cost = 1,9017
Epoch 105, Update 2000, Cost = 3,0132
Epoch 110, Update 2100, Cost = 2,9038
Epoch 115, Update 2200, Cost = 2,0760
Epoch 121, Update 2300, Cost = 3,7624
Epoch 126, Update 2400, Cost = 2,8318
Epoch 131, Update 2500, Cost = 2,4387
Epoch 136, Update 2600, Cost = 1,8092
Epoch 142, Update 2700, Cost = 3,2085
Epoch 147, Update 2800, Cost = 2,6596
Epoch 152, Update 2900, Cost = 2,2898
Epoch 157, Update 3000, Cost = 1,8082
Epoch 163, Update 3100, Cost = 3,0456
Epoch 168, Update 3200, Cost = 2,6390
Epoch 173, Update 3300, Cost = 2,1394
Epoch 178, Update 3400, Cost = 1,7817
Epoch 184, Update 3500, Cost = 2,9494
Epoch 189, Update 3600, Cost = 2,5568
Epoch 194, Update 3700, Cost = 2,0092
Epoch 199, Update 3800, Cost = 1,7010
Epoch 205, Update 3900, Cost = 2,7874
Epoch 210, Update 4000, Cost = 2,6260
Epoch 215, Update 4100, Cost = 1,9010
Epoch 221, Update 4200, Cost = 3,7381
Epoch 226, Update 4300, Cost = 2,8128
Epoch 231, Update 4400, Cost = 2,4242
Epoch 236, Update 4500, Cost = 1,8015
Epoch 242, Update 4600, Cost = 3,2033
Epoch 247, Update 4700, Cost = 2,6535
Epoch 252, Update 4800, Cost = 2,2846
Epoch 257, Update 4900, Cost = 1,8047
Epoch 263, Update 5000, Cost = 3,0424
Epoch 268, Update 5100, Cost = 2,6389
Epoch 273, Update 5200, Cost = 2,1393
Epoch 278, Update 5300, Cost = 1,7816
Epoch 284, Update 5400, Cost = 2,9494
Epoch 289, Update 5500, Cost = 2,5567
Epoch 294, Update 5600, Cost = 2,0092
Epoch 299, Update 5700, Cost = 1,7010
Epoch 305, Update 5800, Cost = 2,7874
Epoch 310, Update 5900, Cost = 2,6260
Epoch 315, Update 6000, Cost = 1,9010
Epoch 321, Update 6100, Cost = 3,7381
Epoch 326, Update 6200, Cost = 2,8128
Epoch 331, Update 6300, Cost = 2,4242
Epoch 336, Update 6400, Cost = 1,8015
Epoch 342, Update 6500, Cost = 3,2033
Epoch 347, Update 6600, Cost = 2,6535
Epoch 352, Update 6700, Cost = 2,2846
Epoch 357, Update 6800, Cost = 1,8047
Epoch 363, Update 6900, Cost = 3,0424

Translations:
<s> Non lo so </s>
<s> Che cos'è questo ? </s>
<s> Ascolto la radio </s>
<s> Lo capisco </s>
<s> Lui gioca a calcio </s>
             */


            /*
             
Translations:
<s> Il </s>
<s> Dove ? ? ? </s>
<s> Prendo </s>
<s> Il </s>
<s> Il </s>
Epoch 5, Update 100, Cost = 2,6252
Epoch 10, Update 200, Cost = 2,2645

Translations:
<s> Non lo so </s>
<s> Qual è il tuo tuo preferito preferito preferito preferito preferito ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ?
<s> Ascolto la radio </s>
<s> Non lo sì </s>
<s> Lui gioca a </s>
Epoch 5, Update 100, Cost = 3,0516
Epoch 10, Update 200, Cost = 1,8754
Epoch 15, Update 300, Cost = 0,3975

Translations:
<s> Non lo so </s>
<s> Qual è il colore ? ? ? </s>
<s> Ascolto la radio </s>
<s> Non bambini so </s>
<s> Lui gioca a </s>
Epoch 5, Update 100, Cost = 2,3964
Epoch 10, Update 200, Cost = 1,5005
Epoch 15, Update 300, Cost = 7,9250
Epoch 21, Update 400, Cost = 0,8033
Epoch 26, Update 500, Cost = 0,3047
Epoch 31, Update 600, Cost = 0,2156
Epoch 36, Update 700, Cost = 0,1525
Epoch 42, Update 800, Cost = 0,0762
Epoch 47, Update 900, Cost = 0,0878
Epoch 52, Update 1000, Cost = 0,0922

Translations:
<s> Non lo </s>
<s> Qual è il colore colore colore ? ? ? ? </s>
<s> Ascolto la radio </s>
<s> Sto la </s>
<s> Il bambini a calcio calcio </s>
             */


            /*
             Translations:
<s> Il </s>
<s> ? ? </s>
<s> Il </s>
<s> Sto </s>
<s> Il </s>
Epoch 5, Update 100, Cost = 6,7450
Epoch 10, Update 200, Cost = 4,5467

Translations:
<s> Non lo </s>
<s> Qual è il ? ? preferito preferito preferito ? ? ? </s>
<s> Ascolto la radio </s>
<s> Non lo </s>
<s> Mi un </s>
Epoch 5, Update 100, Cost = 3,8757
Epoch 10, Update 200, Cost = 1,7801
Epoch 15, Update 300, Cost = 0,4102

Translations:
<s> Non lo so </s>
<s> Qual è il tuo preferito </s>
<s> Ascolto la radio </s>
<s> Mi lo </s>
<s> Lui gioca a </s>
Epoch 5, Update 100, Cost = 4,0583
Epoch 10, Update 200, Cost = 2,9881
Epoch 15, Update 300, Cost = 7,6581
Epoch 21, Update 400, Cost = 1,1065
Epoch 26, Update 500, Cost = 0,4157
Epoch 31, Update 600, Cost = 0,3590
Epoch 36, Update 700, Cost = 0,2651
Epoch 42, Update 800, Cost = 0,1049
Epoch 47, Update 900, Cost = 0,1371
Epoch 52, Update 1000, Cost = 0,1466

Translations:
<s> Non lo </s>
<s> Qual è il tuo colore ? ? </s>
<s> Ascolto la radio </s>
<s> Ascolto la </s>
<s> Stai attento colore profumo </s>
Epoch 5, Update 100, Cost = 5,5985
Epoch 10, Update 200, Cost = 5,1043
Epoch 15, Update 300, Cost = 1,7284
Epoch 21, Update 400, Cost = 0,1414
Epoch 26, Update 500, Cost = 0,1241
Epoch 31, Update 600, Cost = 0,1148
Epoch 36, Update 700, Cost = 0,0897
Epoch 42, Update 800, Cost = 0,0376
Epoch 47, Update 900, Cost = 0,0523
Epoch 52, Update 1000, Cost = 0,0550
Epoch 57, Update 1100, Cost = 3,8415
Epoch 63, Update 1200, Cost = 4,4116

Translations:
<s> Non lo so </s>
<s> Qual è il ? ? </s>
<s> Ascolto la radio </s>
<s> Lo capisco </s>
<s> Lui gioca a </s>
Epoch 5, Update 100, Cost = 4,4465
Epoch 10, Update 200, Cost = 1,2587
Epoch 15, Update 300, Cost = 0,3906
Epoch 21, Update 400, Cost = 0,0713
Epoch 26, Update 500, Cost = 0,0871
Epoch 31, Update 600, Cost = 0,0711
Epoch 36, Update 700, Cost = 0,0538
Epoch 42, Update 800, Cost = 9,3503
Epoch 47, Update 900, Cost = 7,9235
Epoch 52, Update 1000, Cost = 6,9584
Epoch 57, Update 1100, Cost = 6,5306
Epoch 63, Update 1200, Cost = 4,9911
Epoch 68, Update 1300, Cost = 5,2500
Epoch 73, Update 1400, Cost = 4,9846
Epoch 78, Update 1500, Cost = 5,4906
Epoch 84, Update 1600, Cost = 4,1448

Translations:
<s> Non lo </s>
<s> Che cos'è questo </s>
<s> Sto la radio </s>
<s> Sto capisco </s>
<s> Mi un viaggiare </s>
Epoch 5, Update 100, Cost = 2,2070
Epoch 10, Update 200, Cost = 0,7015
Epoch 15, Update 300, Cost = 0,1758
Epoch 21, Update 400, Cost = 0,0416
Epoch 26, Update 500, Cost = 0,0508
Epoch 31, Update 600, Cost = 0,0477
Epoch 36, Update 700, Cost = 0,0377
Epoch 42, Update 800, Cost = 0,0157
Epoch 47, Update 900, Cost = 0,0261
Epoch 52, Update 1000, Cost = 12,4227
Epoch 57, Update 1100, Cost = 12,2059
Epoch 63, Update 1200, Cost = 8,9593
Epoch 68, Update 1300, Cost = 9,8162
Epoch 73, Update 1400, Cost = 9,2751
Epoch 78, Update 1500, Cost = 10,1790
Epoch 84, Update 1600, Cost = 7,5395
Epoch 89, Update 1700, Cost = 8,9961
Epoch 94, Update 1800, Cost = 8,6442
Epoch 99, Update 1900, Cost = 9,6786
Epoch 105, Update 2000, Cost = 7,4386

Translations:
<s> Lo lo ? </s>
<s> Prego il ? </s>
<s> Sto la </s>
<s> Mi anni </s>
<s> Il venti parco </s>
Epoch 5, Update 100, Cost = 3,9799
Epoch 10, Update 200, Cost = 1,6247
Epoch 15, Update 300, Cost = 0,5977
Epoch 21, Update 400, Cost = 0,0912
Epoch 26, Update 500, Cost = 0,1105
Epoch 31, Update 600, Cost = 0,0867
Epoch 36, Update 700, Cost = 14,0968
Epoch 42, Update 800, Cost = 6,8969
Epoch 47, Update 900, Cost = 6,0976
Epoch 52, Update 1000, Cost = 5,8387
Epoch 57, Update 1100, Cost = 5,6301
Epoch 63, Update 1200, Cost = 4,3519
Epoch 68, Update 1300, Cost = 4,2146
Epoch 73, Update 1400, Cost = 4,1353
Epoch 78, Update 1500, Cost = 4,8500
Epoch 84, Update 1600, Cost = 3,2315
Epoch 89, Update 1700, Cost = 6,5698
Epoch 94, Update 1800, Cost = 6,1292
Epoch 99, Update 1900, Cost = 6,1224
Epoch 105, Update 2000, Cost = 4,3339
Epoch 110, Update 2100, Cost = 6,0150
Epoch 115, Update 2200, Cost = 5,6522
Epoch 121, Update 2300, Cost = 5,9433
Epoch 126, Update 2400, Cost = 4,3764
Epoch 131, Update 2500, Cost = 5,6988
Epoch 136, Update 2600, Cost = 5,3673
Epoch 142, Update 2700, Cost = 4,7690
Epoch 147, Update 2800, Cost = 4,8005
Epoch 152, Update 2900, Cost = 5,8007
Epoch 157, Update 3000, Cost = 5,6498
Epoch 163, Update 3100, Cost = 4,8704
Epoch 168, Update 3200, Cost = 5,7095
Epoch 173, Update 3300, Cost = 5,7161
Epoch 178, Update 3400, Cost = 5,9681
Epoch 184, Update 3500, Cost = 4,2776
Epoch 189, Update 3600, Cost = 5,8697
Epoch 194, Update 3700, Cost = 5,6390
Epoch 199, Update 3800, Cost = 5,8134

Translations:
<s> Non lo so </s>
<s> Che cos'è questo ? ? </s>
<s> Sto la radio </s>
<s> Sto capisco </s>
<s> I gioca a a calcio calcio </s>
Epoch 5, Update 100, Cost = 3,9049
Epoch 10, Update 200, Cost = 0,8424
Epoch 15, Update 300, Cost = 0,3237
Epoch 21, Update 400, Cost = 0,0774
Epoch 26, Update 500, Cost = 0,0882
Epoch 31, Update 600, Cost = 15,1538
Epoch 36, Update 700, Cost = 7,5701
Epoch 42, Update 800, Cost = 4,3009
Epoch 47, Update 900, Cost = 3,1683
Epoch 52, Update 1000, Cost = 2,8039
Epoch 57, Update 1100, Cost = 2,5023
Epoch 63, Update 1200, Cost = 1,8645
Epoch 68, Update 1300, Cost = 1,7903
Epoch 73, Update 1400, Cost = 1,8186
Epoch 78, Update 1500, Cost = 5,2644
Epoch 84, Update 1600, Cost = 3,0429
Epoch 89, Update 1700, Cost = 5,0552
Epoch 94, Update 1800, Cost = 4,6588
Epoch 99, Update 1900, Cost = 4,2364
Epoch 105, Update 2000, Cost = 2,8598
Epoch 110, Update 2100, Cost = 4,5744
Epoch 115, Update 2200, Cost = 4,2535
Epoch 121, Update 2300, Cost = 4,8355
Epoch 126, Update 2400, Cost = 2,7887
Epoch 131, Update 2500, Cost = 4,2133
Epoch 136, Update 2600, Cost = 3,8505
Epoch 142, Update 2700, Cost = 3,5313
Epoch 147, Update 2800, Cost = 3,7018
Epoch 152, Update 2900, Cost = 4,5128
Epoch 157, Update 3000, Cost = 3,9669
Epoch 163, Update 3100, Cost = 3,0491
Epoch 168, Update 3200, Cost = 4,3002
Epoch 173, Update 3300, Cost = 4,3846
Epoch 178, Update 3400, Cost = 4,0098
Epoch 184, Update 3500, Cost = 2,7351
Epoch 189, Update 3600, Cost = 4,5268
Epoch 194, Update 3700, Cost = 4,2120
Epoch 199, Update 3800, Cost = 3,9007
Epoch 205, Update 3900, Cost = 2,7310
Epoch 210, Update 4000, Cost = 4,3214
Epoch 215, Update 4100, Cost = 4,0245
Epoch 221, Update 4200, Cost = 4,7540
Epoch 226, Update 4300, Cost = 2,7785
Epoch 231, Update 4400, Cost = 4,1951
Epoch 236, Update 4500, Cost = 3,8382
Epoch 242, Update 4600, Cost = 3,5290
Epoch 247, Update 4700, Cost = 3,6960
Epoch 252, Update 4800, Cost = 4,5055

Translations:
<s> Non lo so </s>
<s> Che cos'è questo ? </s>
<s> Ascolto la radio </s>
<s> Lo capisco </s>
<s> Lui gioca a a a a a calcio calcio calcio calcio calcio calcio calcio calcio calcio calcio calcio a a a a a calcio calcio calcio calcio calcio a a a a calcio calcio calcio a calcio a calcio a calcio a calcio a calcio a calcio calcio a calcio a calcio a calcio a calcio a calcio a calcio a calcio a
             */


            Console.ReadLine();
        }
    }
}
