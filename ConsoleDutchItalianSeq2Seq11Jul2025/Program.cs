using Seq2SeqSharp;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Enums;
using Seq2SeqSharp.LearningRate;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;

// ChatGPT, modified
namespace ConsoleDutchItalianSeq2Seq11Jul2025
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

                // 52 sentence-pairs
                var trainData = new List<(string src, string tgt)>
                {
                    ("Hallo , hoe gaat het ?", "Ciao , come stai ?"),
                    ("Ik hou van jou", "Ti amo"),
                    ("Dit is een boek", "Questo è un libro"),
                    ("Zij is een lerares", "Lei è un'insegnante"),
                    ("Wat is jouw naam ?", "Come ti chiami ?"),
                    ("Dit is mijn huis", "Questa è la mia casa"),
                    ("Ik spreek een beetje Engels", "Parlo un po' di inglese"),
                    ("Waar is het station ?", "Dove è la stazione ?"),
                    ("Hoe laat is het ?", "Che ore sono ?"),
                    ("Ik ben moe", "Sono stanco"),
                    ("Wat doe je ?", "Cosa fai ?"),
                    ("Ik kom uit Nederland", "Vengo dai Paesi Bassi"),
                    ("Ik heb honger", "Ho fame"),
                    ("Ik heb dorst", "Ho sete"),
                    ("Waar woon je ?", "Dove vivi ?"),
                    ("Ik werk als ingenieur", "Lavoro come ingegnere"),
                    ("Zij is mijn zus", "Lei è mia sorella"),
                    ("Hij is mijn broer", "Lui è mio fratello"),
                    ("Ik wil koffie", "Voglio del caffè"),
                    ("Wil je thee ?", "Vuoi del tè ?"),
                    ("Ik ga naar school", "Vado a scuola"),
                    ("Het is koud vandaag", "Fa freddo oggi"),
                    ("Het regent buiten", "Sta piovendo fuori"),
                    ("De zon schijnt", "Il sole splende"),
                    ("Ik ben blij", "Sono felice"),
                    ("Ik ben verdrietig", "Sono triste"),
                    ("Ik begrijp het niet", "Non lo capisco"),
                    ("Kun je dat herhalen ?", "Puoi ripetere ?"),
                    ("Ik ben ziek", "Sono malato"),
                    ("Waar is de wc ?", "Dove è il bagno ?"),
                    ("Ik moet gaan", "Devo andare"),
                    ("Tot ziens", "Arrivederci"),
                    ("Goede morgen", "Buongiorno"),
                    ("Goede nacht", "Buonanotte"),
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
                    ("De kat slaapt", "Il gatto dorme")
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
                     "Goede morgen",
                     "Ik hou van muziek",
                     "Ik ben ziek",
                     "Dat is mooi",
                     "Waar woon je ?",
                     "De kat wil naar huis",
                     "Ik speel voetbal"
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
<s> </s>
<s> Amo la </s>
<s> Sono </s>
<s> È bello </s>
<s> Dove ? ? ? ? </s>
<s> Voglio casa casa </s>
<s> Lui a a </s>
Epoch 3, Update 100, Cost = 8,1402
Epoch 7, Update 200, Cost = 2,6673
Epoch 11, Update 300, Cost = 14,8895
Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Quanti anni hai ? </s>
<s> Voglio andare a casa </s>
<s> Suono andare a </s>
Epoch 3, Update 100, Cost = 7,2947
Epoch 7, Update 200, Cost = 2,2552
Epoch 11, Update 300, Cost = 0,7517
Epoch 15, Update 400, Cost = 7,8717
Epoch 19, Update 500, Cost = 3,0339
Translations:
<s> Ho </s>
<s> Amo la musica </s>
<s> Sono </s>
<s> È bello </s>
<s> Dove anni ? </s>
<s> Voglio andare a casa </s>
<s> Ho </s>
Epoch 3, Update 100, Cost = 6,6187
Epoch 7, Update 200, Cost = 14,8925
Epoch 11, Update 300, Cost = 3,1047
Epoch 15, Update 400, Cost = 1,3270
Epoch 19, Update 500, Cost = 0,5730
Epoch 23, Update 600, Cost = 2,0963
Epoch 27, Update 700, Cost = 1,0464
Epoch 31, Update 800, Cost = 0,6986
Epoch 35, Update 900, Cost = 0,6848
Epoch 39, Update 1000, Cost = 0,4701
Epoch 43, Update 1100, Cost = 0,4078
Epoch 47, Update 1200, Cost = 0,3702
Epoch 51, Update 1300, Cost = 0,3437
Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Voglio andare a </s>
<s> Suono la chitarra </s>
Epoch 3, Update 100, Cost = 6,3044
Epoch 7, Update 200, Cost = 2,9674
Epoch 11, Update 300, Cost = 6,1911
Epoch 15, Update 400, Cost = 1,4570
Epoch 19, Update 500, Cost = 0,6414
Epoch 23, Update 600, Cost = 1,4779
Epoch 27, Update 700, Cost = 1,5428
Epoch 31, Update 800, Cost = 0,9450
Epoch 35, Update 900, Cost = 0,6843
Epoch 39, Update 1000, Cost = 0,6230
Epoch 43, Update 1100, Cost = 0,4705
Epoch 47, Update 1200, Cost = 0,4199
Epoch 51, Update 1300, Cost = 0,3893
Epoch 55, Update 1400, Cost = 0,3673
Epoch 59, Update 1500, Cost = 0,3534
Epoch 63, Update 1600, Cost = 0,3449
Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Voglio andare a casa </s>
<s> Ho andare </s>
Epoch 3, Update 100, Cost = 9,4841
Epoch 7, Update 200, Cost = 4,0174
Epoch 11, Update 300, Cost = 1,2342
Epoch 15, Update 400, Cost = 5,5492
Epoch 19, Update 500, Cost = 1,6073
Epoch 23, Update 600, Cost = 0,8641
Epoch 27, Update 700, Cost = 0,5778
Epoch 31, Update 800, Cost = 0,6051
Epoch 35, Update 900, Cost = 0,4669
Epoch 39, Update 1000, Cost = 0,3939
Epoch 43, Update 1100, Cost = 0,3372
Epoch 47, Update 1200, Cost = 0,3084
Epoch 51, Update 1300, Cost = 0,2914
Epoch 55, Update 1400, Cost = 0,2804
Epoch 59, Update 1500, Cost = 0,2723
Epoch 63, Update 1600, Cost = 0,2675
Epoch 67, Update 1700, Cost = 0,2643
Epoch 71, Update 1800, Cost = 0,2620
Epoch 75, Update 1900, Cost = 0,2605
Epoch 79, Update 2000, Cost = 0,2596
Epoch 83, Update 2100, Cost = 0,2589
Epoch 87, Update 2200, Cost = 0,2584
Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Voglio andare a casa </s>
<s> Suono la </s>
          */

            /*
    
Translations:
<s> Sono </s>
<s> Sono </s>
<s> Sono </s>
<s> Sono </s>
<s> Dove ? </s>
<s> Sono </s>
<s> Sono </s>
Epoch 3, Update 100, Cost = 12,8382
Epoch 7, Update 200, Cost = 6,5579
Epoch 11, Update 300, Cost = 3,5228

Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Quanti anni hai ? </s>
<s> Voglio andare a </s>
<s> Lui la </s>
Epoch 3, Update 100, Cost = 11,7753
Epoch 7, Update 200, Cost = 3,2438
Epoch 11, Update 300, Cost = 1,2684
Epoch 15, Update 400, Cost = 0,6630
Epoch 19, Update 500, Cost = 0,4033

Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Quanti anni hai ? </s>
<s> Voglio andare a casa </s>
<s> Lui gioca a </s>
Epoch 3, Update 100, Cost = 13,3395
Epoch 7, Update 200, Cost = 8,2410
Epoch 11, Update 300, Cost = 4,4677
Epoch 15, Update 400, Cost = 2,5108
Epoch 19, Update 500, Cost = 1,2330
Epoch 23, Update 600, Cost = 0,7806
Epoch 27, Update 700, Cost = 0,5372
Epoch 31, Update 800, Cost = 0,4123
Epoch 35, Update 900, Cost = 0,3379
Epoch 39, Update 1000, Cost = 8,1769
Epoch 43, Update 1100, Cost = 5,1890
Epoch 47, Update 1200, Cost = 4,2952
Epoch 51, Update 1300, Cost = 3,8558

Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È è </s>
<s> Dove ripetere ? </s>
<s> Voglio andare a casa </s>
<s> Lui la a </s>
Epoch 3, Update 100, Cost = 11,4796
Epoch 7, Update 200, Cost = 5,7889
Epoch 11, Update 300, Cost = 1,7112
Epoch 15, Update 400, Cost = 0,9120
Epoch 19, Update 500, Cost = 0,5011
Epoch 23, Update 600, Cost = 10,5436
Epoch 27, Update 700, Cost = 6,9753
Epoch 31, Update 800, Cost = 5,1232
Epoch 35, Update 900, Cost = 4,0706
Epoch 39, Update 1000, Cost = 3,4481
Epoch 43, Update 1100, Cost = 3,0671
Epoch 47, Update 1200, Cost = 2,8271
Epoch 51, Update 1300, Cost = 2,6725
Epoch 55, Update 1400, Cost = 2,5713
Epoch 59, Update 1500, Cost = 4,9010
Epoch 63, Update 1600, Cost = 4,6689

Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Voglio andare a </s>
<s> Studio la </s>
Epoch 3, Update 100, Cost = 7,0464
Epoch 7, Update 200, Cost = 2,0328
Epoch 11, Update 300, Cost = 0,8070
Epoch 15, Update 400, Cost = 0,3801
Epoch 19, Update 500, Cost = 0,1973
Epoch 23, Update 600, Cost = 0,1553
Epoch 27, Update 700, Cost = 0,0990
Epoch 31, Update 800, Cost = 11,7532
Epoch 35, Update 900, Cost = 9,4953
Epoch 39, Update 1000, Cost = 8,2442
Epoch 43, Update 1100, Cost = 7,4832
Epoch 47, Update 1200, Cost = 6,9995
Epoch 51, Update 1300, Cost = 6,6847
Epoch 55, Update 1400, Cost = 6,4771
Epoch 59, Update 1500, Cost = 6,3389
Epoch 63, Update 1600, Cost = 6,2462
Epoch 67, Update 1700, Cost = 8,4105
Epoch 71, Update 1800, Cost = 8,1590
Epoch 75, Update 1900, Cost = 8,0103
Epoch 79, Update 2000, Cost = 7,9157
Epoch 83, Update 2100, Cost = 7,8539
Epoch 87, Update 2200, Cost = 7,8129

Translations:
<s> Buongiorno </s>
<s> Dove la musica </s>
<s> Il </s>
<s> Vuoi bello </s>
<s> Quanti anni ? </s>
<s> Il a a </s>
<s> Dove è la </s>
Epoch 3, Update 100, Cost = 8,2347
Epoch 7, Update 200, Cost = 2,7224
Epoch 11, Update 300, Cost = 0,7787
Epoch 15, Update 400, Cost = 9,9070
Epoch 19, Update 500, Cost = 3,8722
Epoch 23, Update 600, Cost = 1,8257
Epoch 27, Update 700, Cost = 1,0579
Epoch 31, Update 800, Cost = 0,7210
Epoch 35, Update 900, Cost = 0,5482
Epoch 39, Update 1000, Cost = 0,4515
Epoch 43, Update 1100, Cost = 0,3941
Epoch 47, Update 1200, Cost = 0,3588
Epoch 51, Update 1300, Cost = 3,1176
Epoch 55, Update 1400, Cost = 2,6440
Epoch 59, Update 1500, Cost = 2,4494
Epoch 63, Update 1600, Cost = 2,3340
Epoch 67, Update 1700, Cost = 2,2609
Epoch 71, Update 1800, Cost = 2,2133
Epoch 75, Update 1900, Cost = 2,1818
Epoch 79, Update 2000, Cost = 2,1607
Epoch 83, Update 2100, Cost = 2,1464
Epoch 87, Update 2200, Cost = 2,1368
Epoch 91, Update 2300, Cost = 1,9825
Epoch 95, Update 2400, Cost = 1,9786
Epoch 99, Update 2500, Cost = 1,9759
Epoch 103, Update 2600, Cost = 1,9741
Epoch 107, Update 2700, Cost = 1,9729

Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Il gatto a casa </s>
<s> Suono la chitarra </s>
             */

            /*
             Translations:
<s> </s>
<s> Amo la </s>
<s> Sono </s>
<s> Lui è </s>
<s> Quanti ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ?
<s> Voglio </s>
<s> Sono a </s>
Epoch 3, Update 100, Cost = 8,0865
Epoch 7, Update 200, Cost = 2,9960
Epoch 11, Update 300, Cost = 1,4133

Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Quanti anni hai ? </s>
<s> Voglio andare a casa </s>
<s> Lui gioca a </s>
Epoch 3, Update 100, Cost = 7,9522
Epoch 7, Update 200, Cost = 3,1988
Epoch 11, Update 300, Cost = 1,0605
Epoch 15, Update 400, Cost = 0,6515
Epoch 19, Update 500, Cost = 0,3284

Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Quanti anni hai ? ? </s>
<s> Voglio andare a casa </s>
<s> Lui gioca a </s>
Epoch 3, Update 100, Cost = 14,0163
Epoch 7, Update 200, Cost = 7,6739
Epoch 11, Update 300, Cost = 4,2209
Epoch 15, Update 400, Cost = 1,5997
Epoch 19, Update 500, Cost = 0,8759
Epoch 23, Update 600, Cost = 0,5331
Epoch 27, Update 700, Cost = 0,3657
Epoch 31, Update 800, Cost = 0,2773
Epoch 35, Update 900, Cost = 0,2249
Epoch 39, Update 1000, Cost = 7,5183
Epoch 43, Update 1100, Cost = 5,4636
Epoch 47, Update 1200, Cost = 4,7347
Epoch 51, Update 1300, Cost = 4,3545

Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Il è a </s>
<s> Suono la chitarra </s>
Epoch 3, Update 100, Cost = 11,0838
Epoch 7, Update 200, Cost = 4,2145
Epoch 11, Update 300, Cost = 1,7102
Epoch 15, Update 400, Cost = 0,8727
Epoch 19, Update 500, Cost = 0,5662
Epoch 23, Update 600, Cost = 10,8182
Epoch 27, Update 700, Cost = 7,1666
Epoch 31, Update 800, Cost = 5,2814
Epoch 35, Update 900, Cost = 4,2143
Epoch 39, Update 1000, Cost = 3,5800
Epoch 43, Update 1100, Cost = 3,1898
Epoch 47, Update 1200, Cost = 2,9432
Epoch 51, Update 1300, Cost = 2,7842
Epoch 55, Update 1400, Cost = 2,6800
Epoch 59, Update 1500, Cost = 5,8971
Epoch 63, Update 1600, Cost = 5,6790

Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono felice </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Il è la </s>
<s> Ho la </s>
Epoch 3, Update 100, Cost = 5,9442
Epoch 7, Update 200, Cost = 1,6963
Epoch 11, Update 300, Cost = 1,0181
Epoch 15, Update 400, Cost = 0,4012
Epoch 19, Update 500, Cost = 0,2316
Epoch 23, Update 600, Cost = 0,1542
Epoch 27, Update 700, Cost = 0,1068
Epoch 31, Update 800, Cost = 12,1781
Epoch 35, Update 900, Cost = 10,2430
Epoch 39, Update 1000, Cost = 9,1250
Epoch 43, Update 1100, Cost = 8,4207
Epoch 47, Update 1200, Cost = 7,9649
Epoch 51, Update 1300, Cost = 7,6658
Epoch 55, Update 1400, Cost = 7,4673
Epoch 59, Update 1500, Cost = 7,3347
Epoch 63, Update 1600, Cost = 7,2455
Epoch 67, Update 1700, Cost = 9,9001
Epoch 71, Update 1800, Cost = 9,6767
Epoch 75, Update 1900, Cost = 9,5440
Epoch 79, Update 2000, Cost = 9,4597
Epoch 83, Update 2100, Cost = 9,4048
Epoch 87, Update 2200, Cost = 9,3685

Translations:
<s> Sono </s>
<s> Sono la </s>
<s> Sono </s>
<s> Lui è la </s>
<s> Dove è ? </s>
<s> Sono è </s>
<s> Sono </s>
Epoch 3, Update 100, Cost = 9,8956
Epoch 7, Update 200, Cost = 2,8867
Epoch 11, Update 300, Cost = 0,8703
Epoch 15, Update 400, Cost = 11,0471
Epoch 19, Update 500, Cost = 4,9548
Epoch 23, Update 600, Cost = 2,4913
Epoch 27, Update 700, Cost = 1,5038
Epoch 31, Update 800, Cost = 1,0452
Epoch 35, Update 900, Cost = 0,8060
Epoch 39, Update 1000, Cost = 0,6712
Epoch 43, Update 1100, Cost = 0,5911
Epoch 47, Update 1200, Cost = 0,5416
Epoch 51, Update 1300, Cost = 3,9158
Epoch 55, Update 1400, Cost = 3,4522
Epoch 59, Update 1500, Cost = 3,2748
Epoch 63, Update 1600, Cost = 3,1677
Epoch 67, Update 1700, Cost = 3,0991
Epoch 71, Update 1800, Cost = 3,0541
Epoch 75, Update 1900, Cost = 3,0243
Epoch 79, Update 2000, Cost = 3,0042
Epoch 83, Update 2100, Cost = 2,9907
Epoch 87, Update 2200, Cost = 2,9815
Epoch 91, Update 2300, Cost = 2,8141
Epoch 95, Update 2400, Cost = 2,8101
Epoch 99, Update 2500, Cost = 2,8074
Epoch 103, Update 2600, Cost = 2,8055
Epoch 107, Update 2700, Cost = 2,8043

Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Il del dorme </s>
<s> Voglio andare a </s>
Epoch 3, Update 100, Cost = 7,5863
Epoch 7, Update 200, Cost = 2,1090
Epoch 11, Update 300, Cost = 0,8425
Epoch 15, Update 400, Cost = 15,4174
Epoch 19, Update 500, Cost = 5,8094
Epoch 23, Update 600, Cost = 3,0277
Epoch 27, Update 700, Cost = 1,7827
Epoch 31, Update 800, Cost = 1,2244
Epoch 35, Update 900, Cost = 0,9307
Epoch 39, Update 1000, Cost = 0,7659
Epoch 43, Update 1100, Cost = 0,6685
Epoch 47, Update 1200, Cost = 0,6085
Epoch 51, Update 1300, Cost = 0,5706
Epoch 55, Update 1400, Cost = 5,1114
Epoch 59, Update 1500, Cost = 4,7511
Epoch 63, Update 1600, Cost = 4,5575
Epoch 67, Update 1700, Cost = 4,4403
Epoch 71, Update 1800, Cost = 4,3653
Epoch 75, Update 1900, Cost = 4,3161
Epoch 79, Update 2000, Cost = 4,2832
Epoch 83, Update 2100, Cost = 4,2611
Epoch 87, Update 2200, Cost = 4,2461
Epoch 91, Update 2300, Cost = 4,0416
Epoch 95, Update 2400, Cost = 4,0346
Epoch 99, Update 2500, Cost = 4,0298
Epoch 103, Update 2600, Cost = 4,0266
Epoch 107, Update 2700, Cost = 4,0244
Epoch 111, Update 2800, Cost = 4,0230
Epoch 115, Update 2900, Cost = 4,0223
Epoch 119, Update 3000, Cost = 4,0218
Epoch 123, Update 3100, Cost = 4,0216
Epoch 127, Update 3200, Cost = 4,0170
Epoch 131, Update 3300, Cost = 4,0169
Epoch 135, Update 3400, Cost = 4,0169
Epoch 139, Update 3500, Cost = 4,0169
Epoch 143, Update 3600, Cost = 4,0169
Epoch 147, Update 3700, Cost = 4,0169
Epoch 151, Update 3800, Cost = 4,0169
Epoch 155, Update 3900, Cost = 4,0169
Epoch 159, Update 4000, Cost = 4,0169
Epoch 163, Update 4100, Cost = 4,0169
Epoch 167, Update 4200, Cost = 4,0169
Epoch 171, Update 4300, Cost = 4,0169
Epoch 175, Update 4400, Cost = 4,0169
Epoch 179, Update 4500, Cost = 4,0169
Epoch 183, Update 4600, Cost = 4,0169
Epoch 187, Update 4700, Cost = 4,0169
Epoch 191, Update 4800, Cost = 4,0169
Epoch 195, Update 4900, Cost = 4,0169
Epoch 199, Update 5000, Cost = 4,0169

Translations:
<s> Buonanotte </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Il è a </s>
<s> Suono la chitarra </s>
Epoch 3, Update 100, Cost = 9,6123
Epoch 7, Update 200, Cost = 3,7657
Epoch 11, Update 300, Cost = 1,7641
Epoch 15, Update 400, Cost = 0,8741
Epoch 19, Update 500, Cost = 0,4903
Epoch 23, Update 600, Cost = 0,3186
Epoch 27, Update 700, Cost = 0,2227
Epoch 31, Update 800, Cost = 0,1695
Epoch 35, Update 900, Cost = 0,1387
Epoch 39, Update 1000, Cost = 5,6965
Epoch 43, Update 1100, Cost = 4,4959
Epoch 47, Update 1200, Cost = 3,9003
Epoch 51, Update 1300, Cost = 3,5601
Epoch 55, Update 1400, Cost = 3,3526
Epoch 59, Update 1500, Cost = 3,2211
Epoch 63, Update 1600, Cost = 3,1357
Epoch 67, Update 1700, Cost = 3,0794
Epoch 71, Update 1800, Cost = 3,0418
Epoch 75, Update 1900, Cost = 2,8347
Epoch 79, Update 2000, Cost = 2,8154
Epoch 83, Update 2100, Cost = 2,8024
Epoch 87, Update 2200, Cost = 2,7937
Epoch 91, Update 2300, Cost = 2,7877
Epoch 95, Update 2400, Cost = 2,7837
Epoch 99, Update 2500, Cost = 2,7809
Epoch 103, Update 2600, Cost = 2,7791
Epoch 107, Update 2700, Cost = 2,7778
Epoch 111, Update 2800, Cost = 2,7737
Epoch 115, Update 2900, Cost = 2,7733
Epoch 119, Update 3000, Cost = 2,7730
Epoch 123, Update 3100, Cost = 2,7729
Epoch 127, Update 3200, Cost = 2,7729
Epoch 131, Update 3300, Cost = 2,7729
Epoch 135, Update 3400, Cost = 2,7728
Epoch 139, Update 3500, Cost = 2,7728
Epoch 143, Update 3600, Cost = 2,7728
Epoch 147, Update 3700, Cost = 2,7728
Epoch 151, Update 3800, Cost = 2,7727
Epoch 155, Update 3900, Cost = 2,7727
Epoch 159, Update 4000, Cost = 2,7727
Epoch 163, Update 4100, Cost = 2,7727
Epoch 167, Update 4200, Cost = 2,7727
Epoch 171, Update 4300, Cost = 2,7727
Epoch 175, Update 4400, Cost = 2,7727
Epoch 179, Update 4500, Cost = 2,7727
Epoch 183, Update 4600, Cost = 2,7727
Epoch 187, Update 4700, Cost = 2,7727
Epoch 191, Update 4800, Cost = 2,7727
Epoch 195, Update 4900, Cost = 2,7727
Epoch 199, Update 5000, Cost = 2,7727
Epoch 203, Update 5100, Cost = 2,7727
Epoch 207, Update 5200, Cost = 2,7727
Epoch 211, Update 5300, Cost = 2,7727
Epoch 215, Update 5400, Cost = 2,7727
Epoch 219, Update 5500, Cost = 2,7727
Epoch 223, Update 5600, Cost = 2,7727
Epoch 227, Update 5700, Cost = 2,7727
Epoch 231, Update 5800, Cost = 2,7727
Epoch 235, Update 5900, Cost = 2,7727
Epoch 239, Update 6000, Cost = 2,7727
Epoch 243, Update 6100, Cost = 2,7727
Epoch 247, Update 6200, Cost = 2,7727
Epoch 251, Update 6300, Cost = 2,7727

Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Quanti anni ? ? </s>
<s> Il gatto dorme </s>
<s> Suono la chitarra </s>
             */

            Console.ReadLine();
        }
    }
}
