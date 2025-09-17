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
namespace ConsoleDutchItalianSeq2Seq13Jul2025
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
Translations:
<s> Sono </s>
<s> Sono </s>
<s> Sono </s>
<s> Sono </s>
<s> Dove ? ? </s>
<s> Sono </s>
<s> Sono </s>
Epoch 3, Update 100, Cost = 14,2222
Epoch 6, Update 200, Cost = 6,4216
Epoch 10, Update 300, Cost = 3,6989
Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Voglio andare a casa </s>
<s> Lui gioca a </s>
Epoch 3, Update 100, Cost = 14,2275
Epoch 6, Update 200, Cost = 9,8222
Epoch 10, Update 300, Cost = 6,2655
Epoch 13, Update 400, Cost = 4,2348
Epoch 17, Update 500, Cost = 3,0056
Translations:
<s> Buongiorno </s>
<s> Ti la </s>
<s> Sono la </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Andiamo al </s>
<s> Ho andare anni </s>
Epoch 3, Update 100, Cost = 16,0989
Epoch 6, Update 200, Cost = 14,6468
Epoch 10, Update 300, Cost = 12,6753
Epoch 13, Update 400, Cost = 9,6164
Epoch 17, Update 500, Cost = 6,7834
Epoch 20, Update 600, Cost = 5,2174
Epoch 24, Update 700, Cost = 3,6782
Epoch 27, Update 800, Cost = 3,2442
Epoch 31, Update 900, Cost = 2,8299
Epoch 34, Update 1000, Cost = 2,3276
Epoch 37, Update 1100, Cost = 1,9994
Epoch 41, Update 1200, Cost = 1,8495
Epoch 44, Update 1300, Cost = 1,7001
Epoch 48, Update 1400, Cost = 1,8374
Epoch 51, Update 1500, Cost = 1,5804
Translations:
<s> Buongiorno </s>
<s> Ti amo musica </s>
<s> Sono stanco </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Lui del a calcio </s>
<s> Voglio andare caffè </s>
 */


            /*
             Translations:
<s> Sono </s>
<s> Sono </s>
<s> Sono </s>
<s> Sono </s>
<s> Sono ? ? </s>
<s> Sono </s>
<s> Sono </s>
Epoch 3, Update 100, Cost = 13,0828
Epoch 6, Update 200, Cost = 6,3705
Epoch 10, Update 300, Cost = 2,9071
Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Voglio andare a casa </s>
<s> Lui gioca a </s>
Epoch 3, Update 100, Cost = 14,9223
Epoch 6, Update 200, Cost = 9,6445
Epoch 10, Update 300, Cost = 6,4061
Epoch 13, Update 400, Cost = 3,9835
Epoch 17, Update 500, Cost = 2,9669
Translations:
<s> Buongiorno </s>
<s> Ho la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Il gatto dorme </s>
<s> Ho la chitarra </s>
Epoch 3, Update 100, Cost = 15,4459
Epoch 6, Update 200, Cost = 12,2259
Epoch 10, Update 300, Cost = 9,2289
Epoch 13, Update 400, Cost = 7,1028
Epoch 17, Update 500, Cost = 5,2978
Epoch 20, Update 600, Cost = 3,3324
Epoch 24, Update 700, Cost = 2,3785
Epoch 27, Update 800, Cost = 1,7361
Epoch 31, Update 900, Cost = 1,3878
Epoch 34, Update 1000, Cost = 1,2063
Epoch 37, Update 1100, Cost = 0,9069
Epoch 41, Update 1200, Cost = 1,0203
Epoch 44, Update 1300, Cost = 0,7954
Epoch 48, Update 1400, Cost = 1,0617
Epoch 51, Update 1500, Cost = 0,7250
Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Il è splende </s>
<s> Suono andare scuola </s>
Epoch 3, Update 100, Cost = 14,5856
Epoch 6, Update 200, Cost = 7,3831
Epoch 10, Update 300, Cost = 4,1736
Epoch 13, Update 400, Cost = 1,0957
Epoch 17, Update 500, Cost = 1,1110
Epoch 20, Update 600, Cost = 0,2852
Epoch 24, Update 700, Cost = 0,4747
Epoch 27, Update 800, Cost = 0,1335
Epoch 31, Update 900, Cost = 0,3153
Epoch 34, Update 1000, Cost = 0,0855
Epoch 37, Update 1100, Cost = 0,0524
Epoch 41, Update 1200, Cost = 0,0725
Epoch 44, Update 1300, Cost = 0,0422
Epoch 48, Update 1400, Cost = 0,0713
Epoch 51, Update 1500, Cost = 0,0388
Epoch 55, Update 1600, Cost = 0,0870
Epoch 58, Update 1700, Cost = 0,0409
Epoch 62, Update 1800, Cost = 0,0826
Epoch 65, Update 1900, Cost = 0,0445
Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Voglio andare a casa </s>
<s> Lui gioca a </s>
Epoch 3, Update 100, Cost = 16,8079
Epoch 6, Update 200, Cost = 12,8108
Epoch 10, Update 300, Cost = 7,8095
Epoch 13, Update 400, Cost = 2,9792
Epoch 17, Update 500, Cost = 2,7234
Epoch 20, Update 600, Cost = 0,8202
Epoch 24, Update 700, Cost = 0,8075
Epoch 27, Update 800, Cost = 0,3115
Epoch 31, Update 900, Cost = 0,7620
Epoch 34, Update 1000, Cost = 0,1777
Epoch 37, Update 1100, Cost = 0,1084
Epoch 41, Update 1200, Cost = 0,1474
Epoch 44, Update 1300, Cost = 0,0857
Epoch 48, Update 1400, Cost = 0,1455
Epoch 51, Update 1500, Cost = 0,0783
Epoch 55, Update 1600, Cost = 0,1768
Epoch 58, Update 1700, Cost = 0,0816
Epoch 62, Update 1800, Cost = 0,1719
Epoch 65, Update 1900, Cost = 0,0869
Epoch 68, Update 2000, Cost = 0,0659
Epoch 72, Update 2100, Cost = 0,0995
Epoch 75, Update 2200, Cost = 0,0672
Epoch 79, Update 2300, Cost = 0,1137
Epoch 82, Update 2400, Cost = 0,0697
Epoch 86, Update 2500, Cost = 0,1484
Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Voglio andare a casa </s>
<s> Lui gioca a </s>
Epoch 3, Update 100, Cost = 12,3325
Epoch 6, Update 200, Cost = 5,6660
Epoch 10, Update 300, Cost = 3,0463
Epoch 13, Update 400, Cost = 0,8946
Epoch 17, Update 500, Cost = 0,5594
Epoch 20, Update 600, Cost = 0,3012
Epoch 24, Update 700, Cost = 0,2733
Epoch 27, Update 800, Cost = 0,1414
Epoch 31, Update 900, Cost = 0,2766
Epoch 34, Update 1000, Cost = 0,0717
Epoch 37, Update 1100, Cost = 0,0637
Epoch 41, Update 1200, Cost = 0,0643
Epoch 44, Update 1300, Cost = 0,0528
Epoch 48, Update 1400, Cost = 0,0637
Epoch 51, Update 1500, Cost = 0,0493
Epoch 55, Update 1600, Cost = 0,0770
Epoch 58, Update 1700, Cost = 0,0490
Epoch 62, Update 1800, Cost = 0,0864
Epoch 65, Update 1900, Cost = 0,0502
Epoch 68, Update 2000, Cost = 0,0417
Epoch 72, Update 2100, Cost = 0,0497
Epoch 75, Update 2200, Cost = 0,0429
Epoch 79, Update 2300, Cost = 0,0538
Epoch 82, Update 2400, Cost = 0,0437
Epoch 86, Update 2500, Cost = 0,0647
Epoch 89, Update 2600, Cost = 0,0449
Epoch 93, Update 2700, Cost = 0,0636
Epoch 96, Update 2800, Cost = 0,0485
Epoch 99, Update 2900, Cost = 0,0410
Epoch 103, Update 3000, Cost = 0,0480
Epoch 106, Update 3100, Cost = 0,0423
Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Voglio andare a casa </s>
<s> Lui gioca a </s>
             */


            /*
       Translations:
<s> Il </s>
<s> Sono </s>
<s> Ho </s>
<s> Lui è </s>
<s> Dove ? ? ? ? ? ? ? ? ? ? ? ? ? </s>
<s> Il Il </s>
<s> Il andare </s>
Epoch 3, Update 100, Cost = 10,8500
Epoch 6, Update 200, Cost = 12,2689
Epoch 10, Update 300, Cost = 4,8626

Translations:
<s> Facciamo un buon </s>
<s> Facciamo una passeggiata </s>
<s> Sto una buon </s>
<s> Il bello film </s>
<s> Cosa vuoi fare </s>
<s> Voglio andare a a </s>
<s> Sto un passeggiata </s>
Epoch 3, Update 100, Cost = 11,0144
Epoch 6, Update 200, Cost = 8,0060
Epoch 10, Update 300, Cost = 9,4635
Epoch 13, Update 400, Cost = 5,2212
Epoch 17, Update 500, Cost = 6,5260

Translations:
<s> Buongiorno </s>
<s> Ho la </s>
<s> Sono la </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Il sole splende </s>
<s> Sono la </s>
Epoch 3, Update 100, Cost = 10,4875
Epoch 6, Update 200, Cost = 10,4217
Epoch 10, Update 300, Cost = 4,1829
Epoch 13, Update 400, Cost = 7,7793
Epoch 17, Update 500, Cost = 4,6156
Epoch 20, Update 600, Cost = 2,5977
Epoch 24, Update 700, Cost = 3,3880
Epoch 27, Update 800, Cost = 1,9203
Epoch 31, Update 900, Cost = 1,5678
Epoch 34, Update 1000, Cost = 1,3447
Epoch 37, Update 1100, Cost = 0,9931
Epoch 41, Update 1200, Cost = 1,1267
Epoch 44, Update 1300, Cost = 0,8611
Epoch 48, Update 1400, Cost = 1,1741
Epoch 51, Update 1500, Cost = 0,7523

Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Il gatto dorme </s>
<s> Suono la chitarra </s>
Epoch 3, Update 100, Cost = 10,4384
Epoch 6, Update 200, Cost = 13,8031
Epoch 10, Update 300, Cost = 3,9687
Epoch 13, Update 400, Cost = 1,7202
Epoch 17, Update 500, Cost = 3,0813
Epoch 20, Update 600, Cost = 0,8977
Epoch 24, Update 700, Cost = 1,3880
Epoch 27, Update 800, Cost = 0,5979
Epoch 31, Update 900, Cost = 0,8019
Epoch 34, Update 1000, Cost = 0,3970
Epoch 37, Update 1100, Cost = 0,2836
Epoch 41, Update 1200, Cost = 0,3273
Epoch 44, Update 1300, Cost = 0,2177
Epoch 48, Update 1400, Cost = 0,3137
Epoch 51, Update 1500, Cost = 0,1947
Epoch 55, Update 1600, Cost = 0,3899
Epoch 58, Update 1700, Cost = 0,2027
Epoch 62, Update 1800, Cost = 0,3892
Epoch 65, Update 1900, Cost = 0,2142

Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Il gatto splende </s>
<s> Suono la chitarra </s>
Epoch 3, Update 100, Cost = 17,8474
Epoch 6, Update 200, Cost = 6,0872
Epoch 10, Update 300, Cost = 4,6535
Epoch 13, Update 400, Cost = 3,2124
Epoch 17, Update 500, Cost = 3,3928
Epoch 20, Update 600, Cost = 1,4254
Epoch 24, Update 700, Cost = 1,8999
Epoch 27, Update 800, Cost = 1,0518
Epoch 31, Update 900, Cost = 1,8025
Epoch 34, Update 1000, Cost = 0,7186
Epoch 37, Update 1100, Cost = 0,4789
Epoch 41, Update 1200, Cost = 0,6965
Epoch 44, Update 1300, Cost = 0,4269
Epoch 48, Update 1400, Cost = 0,7975
Epoch 51, Update 1500, Cost = 0,3953
Epoch 55, Update 1600, Cost = 1,0922
Epoch 58, Update 1700, Cost = 0,4373
Epoch 62, Update 1800, Cost = 0,8740
Epoch 65, Update 1900, Cost = 0,4692
Epoch 68, Update 2000, Cost = 0,3649
Epoch 72, Update 2100, Cost = 0,5509
Epoch 75, Update 2200, Cost = 0,3750
Epoch 79, Update 2300, Cost = 0,6658
Epoch 82, Update 2400, Cost = 0,4046
Epoch 86, Update 2500, Cost = 0,9099

Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Voglio andare a </s>
<s> Ho la </s>
             */


            /*
            Translations:
<s> </s>
<s> </s>
<s> </s>
<s> </s>
<s> </s>
<s> </s>
<s> </s>
Epoch 3, Update 100, Cost = 13,3934
Epoch 6, Update 200, Cost = 13,0367
Epoch 10, Update 300, Cost = 5,9182
Translations:
<s> Il </s>
<s> Sto un </s>
<s> Il un </s>
<s> Sto un film </s>
<s> Cosa vuoi fare ? </s>
<s> Sto andare a a casa casa casa casa casa casa a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a
<s> Sto un </s>
Epoch 3, Update 100, Cost = 12,5409
Epoch 6, Update 200, Cost = 9,0365
Epoch 10, Update 300, Cost = 11,8864
Epoch 13, Update 400, Cost = 5,3935
Epoch 17, Update 500, Cost = 6,2867
Translations:
<s> </s>
<s> Ti la </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Il sole splende </s>
<s> Non la </s>
Epoch 3, Update 100, Cost = 10,1723
Epoch 6, Update 200, Cost = 9,2464
Epoch 10, Update 300, Cost = 3,1758
Epoch 13, Update 400, Cost = 7,9827
Epoch 17, Update 500, Cost = 2,2551
Epoch 20, Update 600, Cost = 0,6959
Epoch 24, Update 700, Cost = 0,8214
Epoch 27, Update 800, Cost = 0,4302
Epoch 31, Update 900, Cost = 0,7049
Epoch 34, Update 1000, Cost = 0,2417
Epoch 37, Update 1100, Cost = 0,1659
Epoch 41, Update 1200, Cost = 0,1912
Epoch 44, Update 1300, Cost = 0,1337
Epoch 48, Update 1400, Cost = 0,1872
Epoch 51, Update 1500, Cost = 0,1204
Translations:
<s> Buonanotte </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Voglio andare a </s>
<s> Suono la </s>
Epoch 3, Update 100, Cost = 8,2615
Epoch 6, Update 200, Cost = 14,5803
Epoch 10, Update 300, Cost = 2,6727
Epoch 13, Update 400, Cost = 2,0936
Epoch 17, Update 500, Cost = 3,4072
Epoch 20, Update 600, Cost = 1,3053
Epoch 24, Update 700, Cost = 1,4773
Epoch 27, Update 800, Cost = 0,6839
Epoch 31, Update 900, Cost = 1,0229
Epoch 34, Update 1000, Cost = 0,4405
Epoch 37, Update 1100, Cost = 0,3081
Epoch 41, Update 1200, Cost = 0,3693
Epoch 44, Update 1300, Cost = 0,2547
Epoch 48, Update 1400, Cost = 0,3820
Epoch 51, Update 1500, Cost = 0,2196
Epoch 55, Update 1600, Cost = 0,4879
Epoch 58, Update 1700, Cost = 0,2327
Epoch 62, Update 1800, Cost = 0,5920
Epoch 65, Update 1900, Cost = 0,2505
Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Il gatto dorme </s>
<s> Suono la chitarra </s> 
             */


            /*
             Translations:
<s> Sono </s>
<s> Sono </s>
<s> Sono </s>
<s> Sono </s>
<s> Sono </s>
<s> Sono </s>
<s> Sono </s>
Epoch 3, Update 100, Cost = 14,0093
Epoch 6, Update 200, Cost = 12,5502
Epoch 10, Update 300, Cost = 4,2907
Translations:
<s> Buonanotte </s>
<s> Sto cucinando un </s>
<s> Sto cucinando un </s>
<s> È bello bello </s>
<s> Dove vuoi hai ? ? </s>
<s> Il cane cane abbaia abbaia </s>
<s> Sto cucinando un </s>
Epoch 3, Update 100, Cost = 10,4639
Epoch 6, Update 200, Cost = 4,4447
Epoch 10, Update 300, Cost = 8,8866
Epoch 13, Update 400, Cost = 2,4614
Epoch 17, Update 500, Cost = 5,4846
Translations:
<s> Buongiorno </s>
<s> Ti la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Dove vivi ? ? </s>
<s> Voglio andare a </s>
<s> Suono la chitarra </s>
Epoch 3, Update 100, Cost = 8,8694
Epoch 6, Update 200, Cost = 12,0280
Epoch 10, Update 300, Cost = 4,3945
Epoch 13, Update 400, Cost = 7,6448
Epoch 17, Update 500, Cost = 4,3994
Epoch 20, Update 600, Cost = 0,9978
Epoch 24, Update 700, Cost = 1,4707
Epoch 27, Update 800, Cost = 0,5115
Epoch 31, Update 900, Cost = 1,3609
Epoch 34, Update 1000, Cost = 0,3296
Epoch 37, Update 1100, Cost = 0,2409
Epoch 41, Update 1200, Cost = 0,2826
Epoch 44, Update 1300, Cost = 0,1893
Epoch 48, Update 1400, Cost = 0,2891
Epoch 51, Update 1500, Cost = 0,1626
Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Voglio andare a casa </s>
<s> Lui gioca a </s>
Epoch 3, Update 100, Cost = 11,4907
Epoch 6, Update 200, Cost = 14,4565
Epoch 10, Update 300, Cost = 3,6514
Epoch 13, Update 400, Cost = 1,7856
Epoch 17, Update 500, Cost = 4,0377
Epoch 20, Update 600, Cost = 1,9532
Epoch 24, Update 700, Cost = 1,1256
Epoch 27, Update 800, Cost = 0,8059
Epoch 31, Update 900, Cost = 0,7139
Epoch 34, Update 1000, Cost = 0,4461
Epoch 37, Update 1100, Cost = 0,3667
Epoch 41, Update 1200, Cost = 0,3400
Epoch 44, Update 1300, Cost = 0,2873
Epoch 48, Update 1400, Cost = 0,3226
Epoch 51, Update 1500, Cost = 0,2420
Epoch 55, Update 1600, Cost = 0,3408
Epoch 58, Update 1700, Cost = 0,2444
Epoch 62, Update 1800, Cost = 0,4104
Epoch 65, Update 1900, Cost = 0,2536
Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Il sole splende </s>
<s> Suono la chitarra </s>
Epoch 3, Update 100, Cost = 17,2702
Epoch 6, Update 200, Cost = 4,5046
Epoch 10, Update 300, Cost = 2,7841
Epoch 13, Update 400, Cost = 2,0297
Epoch 17, Update 500, Cost = 1,1920
Epoch 20, Update 600, Cost = 0,5135
Epoch 24, Update 700, Cost = 0,5941
Epoch 27, Update 800, Cost = 0,2947
Epoch 31, Update 900, Cost = 0,6431
Epoch 34, Update 1000, Cost = 0,1329
Epoch 37, Update 1100, Cost = 0,0780
Epoch 41, Update 1200, Cost = 0,1122
Epoch 44, Update 1300, Cost = 0,0634
Epoch 48, Update 1400, Cost = 0,1048
Epoch 51, Update 1500, Cost = 0,0584
Epoch 55, Update 1600, Cost = 0,1335
Epoch 58, Update 1700, Cost = 0,0613
Epoch 62, Update 1800, Cost = 0,1653
Epoch 65, Update 1900, Cost = 0,0667
Epoch 68, Update 2000, Cost = 0,0487
Epoch 72, Update 2100, Cost = 0,0766
Epoch 75, Update 2200, Cost = 0,0503
Epoch 79, Update 2300, Cost = 0,0826
Epoch 82, Update 2400, Cost = 0,0518
Epoch 86, Update 2500, Cost = 0,1092
Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Voglio andare a casa </s>
<s> Lui gioca a </s>
             */

            Console.ReadLine();
        }
    }
}
