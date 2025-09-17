using Seq2SeqSharp;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Enums;
using Seq2SeqSharp.LearningRate;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;

// ChatGPT, modified
namespace ConsoleDutchItalianSeq2Seq12Jul2025
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

                // 60 sentence-pairs
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
                    ("De kat slaapt", "Il gatto dorme"),
                    ("De hond blaft", "Il cane abbaia"),
                    ("Ik lees een boek", "Sto leggendo un libro"),
                    ("We kijken een film", "Guardiamo un film"),
                    ("Wat wil je doen ?", "Cosa vuoi fare ?"),
                    ("Laten we gaan wandelen", "Facciamo una passeggiata"),
                    ("Ik ben aan het koken", "Sto cucinando"),
                    ("Het ruikt lekker", "Ha un buon profumo"),
                    ("Ben je klaar ?", "Sei pronto ?")
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
<s> Sono </s>
<s> Sono </s>
<s> Il è </s>
<s> Cosa ? ? ? </s>
<s> Il andare </s>
<s> Lui Sto </s>
Epoch 3, Update 100, Cost = 10,0903
Epoch 6, Update 200, Cost = 4,9559
Epoch 10, Update 300, Cost = 9,7095
Translations:
<s> Voglio un </s>
<s> Amo la la </s>
<s> Ti un </s>
<s> Lui è mia </s>
<s> Dove è ? ? </s>
<s> Voglio andare la casa </s>
<s> Ho un </s>
Epoch 3, Update 100, Cost = 11,2839
Epoch 6, Update 200, Cost = 3,4093
Epoch 10, Update 300, Cost = 10,9707
Epoch 13, Update 400, Cost = 6,2063
Epoch 17, Update 500, Cost = 2,4896
Translations:
<s> Ha un film </s>
<s> Amo la musica </s>
<s> Sto cucinando </s>
<s> È un </s>
<s> Dove pronto ? ? </s>
<s> Dove del sono sorella </s>
<s> Sto cucinando musica </s>
Epoch 3, Update 100, Cost = 10,5676
Epoch 6, Update 200, Cost = 5,4649
Epoch 10, Update 300, Cost = 2,7061
Epoch 13, Update 400, Cost = 7,2468
Epoch 17, Update 500, Cost = 4,7387
Epoch 20, Update 600, Cost = 1,7018
Epoch 24, Update 700, Cost = 2,6766
Epoch 27, Update 800, Cost = 1,3896
Epoch 31, Update 900, Cost = 5,2762
Epoch 34, Update 1000, Cost = 1,1232
Epoch 37, Update 1100, Cost = 0,7157
Epoch 41, Update 1200, Cost = 1,1227
Epoch 44, Update 1300, Cost = 0,6773
Epoch 48, Update 1400, Cost = 1,2821
Epoch 51, Update 1500, Cost = 0,6630
Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Voglio andare a casa </s>
<s> Suono la </s>
Epoch 3, Update 100, Cost = 9,3835
Epoch 6, Update 200, Cost = 16,4313
Epoch 10, Update 300, Cost = 7,7527
Epoch 13, Update 400, Cost = 5,1305
Epoch 17, Update 500, Cost = 3,6671
Epoch 20, Update 600, Cost = 5,5456
Epoch 24, Update 700, Cost = 5,4256
Epoch 27, Update 800, Cost = 3,0933
Epoch 31, Update 900, Cost = 5,1203
Epoch 34, Update 1000, Cost = 2,2492
Epoch 37, Update 1100, Cost = 1,9009
Epoch 41, Update 1200, Cost = 2,1748
Epoch 44, Update 1300, Cost = 1,6651
Epoch 48, Update 1400, Cost = 2,1386
Epoch 51, Update 1500, Cost = 1,5884
Epoch 55, Update 1600, Cost = 2,4985
Epoch 58, Update 1700, Cost = 1,5667
Epoch 62, Update 1800, Cost = 2,4968
Epoch 65, Update 1900, Cost = 1,6307
Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Voglio andare a casa </s>
<s> Vengo andare a </s>
Epoch 3, Update 100, Cost = 9,9571
Epoch 6, Update 200, Cost = 5,4640
Epoch 10, Update 300, Cost = 1,9766
Epoch 13, Update 400, Cost = 7,6973
Epoch 17, Update 500, Cost = 2,7707
Epoch 20, Update 600, Cost = 1,6157
Epoch 24, Update 700, Cost = 6,9971
Epoch 27, Update 800, Cost = 2,6602
Epoch 31, Update 900, Cost = 5,3990
Epoch 34, Update 1000, Cost = 1,8041
Epoch 37, Update 1100, Cost = 1,4912
Epoch 41, Update 1200, Cost = 1,7633
Epoch 44, Update 1300, Cost = 1,2608
Epoch 48, Update 1400, Cost = 1,6546
Epoch 51, Update 1500, Cost = 1,1058
Epoch 55, Update 1600, Cost = 1,9983
Epoch 58, Update 1700, Cost = 1,1450
Epoch 62, Update 1800, Cost = 2,3331
Epoch 65, Update 1900, Cost = 1,1923
Epoch 68, Update 2000, Cost = 1,0084
Epoch 72, Update 2100, Cost = 1,2849
Epoch 75, Update 2200, Cost = 1,0413
Epoch 79, Update 2300, Cost = 1,4075
Epoch 82, Update 2400, Cost = 1,0361
Epoch 86, Update 2500, Cost = 1,7217
Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Voglio andare a casa </s>
<s> Sono sete a </s>
           */

            /*
       
Translations:
<s> </s>
<s> Ho </s>
<s> Sono </s>
<s> È è </s>
<s> Cosa ? ? ? </s>
<s> Voglio andare a </s>
<s> Ho </s>
Epoch 3, Update 100, Cost = 10,2365
Epoch 6, Update 200, Cost = 4,0664
Epoch 10, Update 300, Cost = 2,3763

Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Voglio andare a </s>
<s> Suono un </s>
Epoch 3, Update 100, Cost = 10,9284
Epoch 6, Update 200, Cost = 5,0254
Epoch 10, Update 300, Cost = 2,7749
Epoch 13, Update 400, Cost = 1,1962
Epoch 17, Update 500, Cost = 18,3435

Translations:
<s> Sono </s>
<s> Ho a </s>
<s> Sono </s>
<s> Dove è </s>
<s> Dove è ? </s>
<s> Voglio del </s>
<s> Sono a </s>
Epoch 3, Update 100, Cost = 9,8571
Epoch 6, Update 200, Cost = 3,6049
Epoch 10, Update 300, Cost = 1,6297
Epoch 13, Update 400, Cost = 0,7017
Epoch 17, Update 500, Cost = 0,7530
Epoch 20, Update 600, Cost = 0,2837
Epoch 24, Update 700, Cost = 0,4219
Epoch 27, Update 800, Cost = 0,1749
Epoch 31, Update 900, Cost = 19,5902
Epoch 34, Update 1000, Cost = 13,1754
Epoch 37, Update 1100, Cost = 11,7943
Epoch 41, Update 1200, Cost = 11,7508
Epoch 44, Update 1300, Cost = 10,6254
Epoch 48, Update 1400, Cost = 11,1013
Epoch 51, Update 1500, Cost = 9,9213

Translations:
<s> Cosa </s>
<s> Il </s>
<s> Sono </s>
<s> Il è </s>
<s> Dove è ? </s>
<s> Il andare a </s>
<s> Cosa </s>
Epoch 3, Update 100, Cost = 6,9980
Epoch 6, Update 200, Cost = 14,5811
Epoch 10, Update 300, Cost = 3,6919
Epoch 13, Update 400, Cost = 2,6203
Epoch 17, Update 500, Cost = 1,3259
Epoch 20, Update 600, Cost = 0,6983
Epoch 24, Update 700, Cost = 0,6283
Epoch 27, Update 800, Cost = 0,3946
Epoch 31, Update 900, Cost = 0,7659
Epoch 34, Update 1000, Cost = 0,2906
Epoch 37, Update 1100, Cost = 0,2505
Epoch 41, Update 1200, Cost = 8,4522
Epoch 44, Update 1300, Cost = 5,5092
Epoch 48, Update 1400, Cost = 8,2754
Epoch 51, Update 1500, Cost = 5,3495
Epoch 55, Update 1600, Cost = 9,0078
Epoch 58, Update 1700, Cost = 5,5093
Epoch 62, Update 1800, Cost = 10,1813
Epoch 65, Update 1900, Cost = 5,9732

Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono venti </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Voglio andare casa </s>
<s> Sto la </s>
Epoch 3, Update 100, Cost = 17,1219
Epoch 6, Update 200, Cost = 9,0545
Epoch 10, Update 300, Cost = 4,2248
Epoch 13, Update 400, Cost = 2,2869
Epoch 17, Update 500, Cost = 1,4015
Epoch 20, Update 600, Cost = 0,7003
Epoch 24, Update 700, Cost = 0,6563
Epoch 27, Update 800, Cost = 0,3912
Epoch 31, Update 900, Cost = 1,0393
Epoch 34, Update 1000, Cost = 0,2960
Epoch 37, Update 1100, Cost = 5,8930
Epoch 41, Update 1200, Cost = 9,0617
Epoch 44, Update 1300, Cost = 5,2309
Epoch 48, Update 1400, Cost = 9,3154
Epoch 51, Update 1500, Cost = 5,2915
Epoch 55, Update 1600, Cost = 9,7776
Epoch 58, Update 1700, Cost = 5,6999
Epoch 62, Update 1800, Cost = 13,0475
Epoch 65, Update 1900, Cost = 6,3740
Epoch 68, Update 2000, Cost = 3,6153
Epoch 72, Update 2100, Cost = 6,5205
Epoch 75, Update 2200, Cost = 3,9672
Epoch 79, Update 2300, Cost = 7,6351
Epoch 82, Update 2400, Cost = 4,3709
Epoch 86, Update 2500, Cost = 8,0707

Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Cosa vivi ? </s>
<s> Il andare dorme </s>
<s> Suono la chitarra </s>
             */

            /*
             Translations:
<s> </s>
<s> Amo la </s>
<s> Sono </s>
<s> È bello </s>
<s> Cosa ? ? ? ? ? ? </s>
<s> Voglio </s>
<s> Ho a </s>
Epoch 3, Update 100, Cost = 11,7557
Epoch 6, Update 200, Cost = 5,5575
Epoch 10, Update 300, Cost = 3,9396

Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sto cucinando </s>
<s> È bello </s>
<s> Cosa vuoi ? ? </s>
<s> Voglio andare a </s>
<s> Sto cucinando chitarra </s>
Epoch 3, Update 100, Cost = 11,1364
Epoch 6, Update 200, Cost = 4,6528
Epoch 10, Update 300, Cost = 2,7663
Epoch 13, Update 400, Cost = 0,9939
Epoch 17, Update 500, Cost = 11,8771

Translations:
<s> Buonanotte </s>
<s> Sono la </s>
<s> Sono </s>
<s> Sono un </s>
<s> Dove ? ? ? </s>
<s> Che sole a </s>
<s> Vengo dai </s>
Epoch 3, Update 100, Cost = 11,6765
Epoch 6, Update 200, Cost = 5,3293
Epoch 10, Update 300, Cost = 2,2073
Epoch 13, Update 400, Cost = 0,8043
Epoch 17, Update 500, Cost = 1,0145
Epoch 20, Update 600, Cost = 0,3471
Epoch 24, Update 700, Cost = 0,4435
Epoch 27, Update 800, Cost = 13,6096
Epoch 31, Update 900, Cost = 15,6319
Epoch 34, Update 1000, Cost = 10,4760
Epoch 37, Update 1100, Cost = 9,5903
Epoch 41, Update 1200, Cost = 9,1575
Epoch 44, Update 1300, Cost = 8,5620
Epoch 48, Update 1400, Cost = 8,6820
Epoch 51, Update 1500, Cost = 7,8946

Translations:
<s> Buongiorno </s>
<s> Sono la </s>
<s> Sono </s>
<s> Lei è </s>
<s> Cosa è ? </s>
<s> Il è </s>
<s> Sono la </s>
Epoch 3, Update 100, Cost = 10,1847
Epoch 6, Update 200, Cost = 4,4882
Epoch 10, Update 300, Cost = 2,7576
Epoch 13, Update 400, Cost = 1,0506
Epoch 17, Update 500, Cost = 1,2745
Epoch 20, Update 600, Cost = 0,4719
Epoch 24, Update 700, Cost = 0,7214
Epoch 27, Update 800, Cost = 0,3038
Epoch 31, Update 900, Cost = 0,7309
Epoch 34, Update 1000, Cost = 8,1112
Epoch 37, Update 1100, Cost = 3,9526
Epoch 41, Update 1200, Cost = 7,0852
Epoch 44, Update 1300, Cost = 3,5202
Epoch 48, Update 1400, Cost = 7,2026
Epoch 51, Update 1500, Cost = 3,5817
Epoch 55, Update 1600, Cost = 8,8401
Epoch 58, Update 1700, Cost = 3,8992
Epoch 62, Update 1800, Cost = 10,5892
Epoch 65, Update 1900, Cost = 4,0766

Translations:
<s> Prego </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Quanti vivi ? </s>
<s> Voglio andare a casa </s>
<s> Lui gioca a </s>
Epoch 3, Update 100, Cost = 12,4856
Epoch 6, Update 200, Cost = 6,0281
Epoch 10, Update 300, Cost = 3,1306
Epoch 13, Update 400, Cost = 1,0259
Epoch 17, Update 500, Cost = 1,2201
Epoch 20, Update 600, Cost = 0,3970
Epoch 24, Update 700, Cost = 0,5317
Epoch 27, Update 800, Cost = 0,2476
Epoch 31, Update 900, Cost = 17,6581
Epoch 34, Update 1000, Cost = 13,9623
Epoch 37, Update 1100, Cost = 12,6076
Epoch 41, Update 1200, Cost = 12,7123
Epoch 44, Update 1300, Cost = 11,4408
Epoch 48, Update 1400, Cost = 11,8163
Epoch 51, Update 1500, Cost = 10,9144
Epoch 55, Update 1600, Cost = 12,6605
Epoch 58, Update 1700, Cost = 10,9541
Epoch 62, Update 1800, Cost = 17,5892
Epoch 65, Update 1900, Cost = 14,0101
Epoch 68, Update 2000, Cost = 12,2896
Epoch 72, Update 2100, Cost = 14,5834
Epoch 75, Update 2200, Cost = 12,3072
Epoch 79, Update 2300, Cost = 14,5025
Epoch 82, Update 2400, Cost = 12,3048
Epoch 86, Update 2500, Cost = 16,1541

Translations:
<s> </s>
<s> Amo la </s>
<s> </s>
<s> Lavoro del </s>
<s> come come </s>
<s> Questo la la </s>
<s> Il la </s>
Epoch 3, Update 100, Cost = 10,3103
Epoch 6, Update 200, Cost = 11,8561
Epoch 10, Update 300, Cost = 3,7264
Epoch 13, Update 400, Cost = 2,0856
Epoch 17, Update 500, Cost = 1,0663
Epoch 20, Update 600, Cost = 0,5212
Epoch 24, Update 700, Cost = 0,5301
Epoch 27, Update 800, Cost = 0,2865
Epoch 31, Update 900, Cost = 0,6207
Epoch 34, Update 1000, Cost = 0,2100
Epoch 37, Update 1100, Cost = 4,0634
Epoch 41, Update 1200, Cost = 6,3836
Epoch 44, Update 1300, Cost = 3,1275
Epoch 48, Update 1400, Cost = 5,9444
Epoch 51, Update 1500, Cost = 3,0725
Epoch 55, Update 1600, Cost = 7,3815
Epoch 58, Update 1700, Cost = 3,2787
Epoch 62, Update 1800, Cost = 10,2833
Epoch 65, Update 1900, Cost = 3,7068
Epoch 68, Update 2000, Cost = 2,2327
Epoch 72, Update 2100, Cost = 4,1682
Epoch 75, Update 2200, Cost = 2,2805
Epoch 79, Update 2300, Cost = 4,4816
Epoch 82, Update 2400, Cost = 2,5104
Epoch 86, Update 2500, Cost = 6,0197
Epoch 89, Update 2600, Cost = 2,7983
Epoch 93, Update 2700, Cost = 7,4854
Epoch 96, Update 2800, Cost = 3,2524
Epoch 99, Update 2900, Cost = 1,9952
Epoch 103, Update 3000, Cost = 3,8575
Epoch 106, Update 3100, Cost = 2,1893

Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Il gatto dorme </s>
<s> Suono la la </s>
Epoch 3, Update 100, Cost = 8,3282
Epoch 6, Update 200, Cost = 2,6367
Epoch 10, Update 300, Cost = 1,8030
Epoch 13, Update 400, Cost = 0,5938
Epoch 17, Update 500, Cost = 0,7992
Epoch 20, Update 600, Cost = 0,2559
Epoch 24, Update 700, Cost = 19,2546
Epoch 27, Update 800, Cost = 11,2303
Epoch 31, Update 900, Cost = 14,2827
Epoch 34, Update 1000, Cost = 8,7439
Epoch 37, Update 1100, Cost = 8,1473
Epoch 41, Update 1200, Cost = 7,8682
Epoch 44, Update 1300, Cost = 7,3255
Epoch 48, Update 1400, Cost = 7,6200
Epoch 51, Update 1500, Cost = 6,5595
Epoch 55, Update 1600, Cost = 14,3020
Epoch 58, Update 1700, Cost = 9,6610
Epoch 62, Update 1800, Cost = 13,9319
Epoch 65, Update 1900, Cost = 10,2017
Epoch 68, Update 2000, Cost = 9,1128
Epoch 72, Update 2100, Cost = 10,6379
Epoch 75, Update 2200, Cost = 9,2616
Epoch 79, Update 2300, Cost = 10,8776
Epoch 82, Update 2400, Cost = 9,1481
Epoch 86, Update 2500, Cost = 12,8880
Epoch 89, Update 2600, Cost = 9,1502
Epoch 93, Update 2700, Cost = 12,4862
Epoch 96, Update 2800, Cost = 9,5707
Epoch 99, Update 2900, Cost = 8,8502
Epoch 103, Update 3000, Cost = 10,5014
Epoch 106, Update 3100, Cost = 9,0290
Epoch 110, Update 3200, Cost = 10,4027
Epoch 113, Update 3300, Cost = 8,9550
Epoch 117, Update 3400, Cost = 11,6662
Epoch 120, Update 3500, Cost = 9,0516
Epoch 124, Update 3600, Cost = 13,8009
Epoch 127, Update 3700, Cost = 9,3472
Epoch 131, Update 3800, Cost = 18,0433
Epoch 134, Update 3900, Cost = 9,9949
Epoch 137, Update 4000, Cost = 8,9582
Epoch 141, Update 4100, Cost = 10,3294
Epoch 144, Update 4200, Cost = 9,1045
Epoch 148, Update 4300, Cost = 10,9494
Epoch 151, Update 4400, Cost = 8,9318
Epoch 155, Update 4500, Cost = 13,5708
Epoch 158, Update 4600, Cost = 9,2084
Epoch 162, Update 4700, Cost = 13,3990
Epoch 165, Update 4800, Cost = 9,8990
Epoch 168, Update 4900, Cost = 8,8567
Epoch 172, Update 5000, Cost = 10,3658
Epoch 175, Update 5100, Cost = 9,0317
Epoch 179, Update 5200, Cost = 10,6371
Epoch 182, Update 5300, Cost = 8,9328
Epoch 186, Update 5400, Cost = 12,6274
Epoch 189, Update 5500, Cost = 9,1394
Epoch 193, Update 5600, Cost = 12,4752
Epoch 196, Update 5700, Cost = 9,5646
Epoch 199, Update 5800, Cost = 8,8454

Translations:
<s> Il </s>
<s> Ho la </s>
<s> Sono malato </s>
<s> Lei è a </s>
<s> Dove ripetere ? </s>
<s> Lei è a </s>
<s> Ho un </s>
             */

            Console.ReadLine();
        }
    }
}
