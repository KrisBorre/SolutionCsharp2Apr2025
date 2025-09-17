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
namespace ConsoleDutchItalianSeq2Seq14Jul2025
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
                    ("Hallo , hoe gaat het ?", "Ciao , come stai ?"),
                    ("Ik hou van jou", "Ti amo"),
                    ("Dit is een boek", "Questo è un libro"),
                    ("Zij is een lerares", "Lei è un'insegnante"),
                    ("Wat is jouw naam ?", "Come ti chiami ?"),
                    ("Dit is mijn huis", "Questa è la mia casa"),
                    ("Ik spreek een beetje Engels", "Parlo un po' di inglese"),
                    ("Waar is het station ?", "Dove è la stazione ?"),
                    ("Hoe laat is het ?", "Che ore sono ?"),
                    ("Ik ben moe", "Sono stanco"), // 10
                    ("Wat doe je ?", "Cosa fai ?"),
                    ("Ik kom uit Nederland", "Vengo dai Paesi Bassi"),
                    ("Ik heb honger", "Ho fame"),
                    ("Ik heb dorst", "Ho sete"),
                    ("Waar woon je ?", "Dove vivi ?"),
                    ("Ik werk als ingenieur", "Lavoro come ingegnere"),
                    ("Zij is mijn zus", "Lei è mia sorella"),
                    ("Hij is mijn broer", "Lui è mio fratello"),
                    ("Ik wil koffie", "Voglio del caffè"),
                    ("Wil je thee ?", "Vuoi del tè ?"), // 20
                    ("Ik ga naar school", "Vado a scuola"),
                    ("Het is koud vandaag", "Fa freddo oggi"),
                    ("Het regent buiten", "Sta piovendo fuori"),
                    ("De zon schijnt", "Il sole splende"),
                    ("Ik ben blij", "Sono felice"),
                    ("Ik ben verdrietig", "Sono triste"),
                    ("Ik begrijp het niet", "Non lo capisco"),
                    ("Kun je dat herhalen ?", "Puoi ripetere ?"),
                    ("Ik ben ziek", "Sono malato"),
                    ("Waar is de wc ?", "Dove è il bagno ?"), // 30
                    ("Ik moet gaan", "Devo andare"),
                    ("Tot ziens", "Arrivederci"),
                    ("Goede morgen", "Buongiorno"),
                    ("Goede nacht", "Buonanotte"),
                    ("Slaap lekker", "Dormi bene"),
                    ("Eet smakelijk", "Buon appetito"),
                    ("Welkom", "Benvenuto"),
                    ("Dank je wel", "Grazie"),
                    ("Alsjeblieft", "Prego"),
                    ("Sorry", "Mi dispiace"), // 40
                    ("Geen probleem", "Nessun problema"),
                    ("Ik begrijp het", "Lo capisco"),
                    ("Hoe oud ben je ?", "Quanti anni hai ?"),
                    ("Ik ben twintig jaar oud", "Ho venti anni"),
                    ("Ik studeer aan de universiteit", "Studio all'università"),
                    ("Wat is dit ?", "Che cos'è questo ?"),
                    ("Dat is mooi", "È bello"),
                    ("Ik hou van muziek", "Amo la musica"),
                    ("Ik speel gitaar", "Suono la chitarra"),
                    ("Hij speelt voetbal", "Lui gioca a calcio"), // 50
                    ("We gaan naar het park", "Andiamo al parco"),
                    ("Ik wil naar huis", "Voglio andare a casa"),
                    ("De kat slaapt", "Il gatto dorme"),
                    ("De hond blaft", "Il cane abbaia"),
                    ("Ik lees een boek", "Sto leggendo un libro"),
                    ("We kijken een film", "Guardiamo un film"),
                    ("Wat wil je doen ?", "Cosa vuoi fare ?"),
                    ("Laten we gaan wandelen", "Facciamo una passeggiata"),
                    ("Ik ben aan het koken", "Sto cucinando"),
                    ("Het ruikt lekker", "Ha un buon profumo"), // 60
                    ("Ben je klaar ?", "Sei pronto ?"),
                    ("Ik weet het niet", "Non lo so"),
                    ("Ik denk van wel", "Penso di sì"),
                    ("Misschien", "Forse"),
                    ("Ik ben bang", "Ho paura"),
                    ("Wees voorzichtig", "Stai attento"),
                    ("Dat is gevaarlijk", "È pericoloso"),
                    ("Ik hou van reizen", "Mi piace viaggiare"),
                    ("Ik ga met de trein", "Prendo il treno"),
                    ("Waar gaan we heen ?", "Dove andiamo ?") // 70
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
                     "De hond slaapt",
                     "Ik speel voetbal",
                     "Ik ga wandelen",
                     "Ik hou van reizen met de trein"
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
<s> Ho </s>
<s> Ho Ho </s>
<s> Ho </s>
<s> Ho è </s>
<s> Dove ? ? ? ? </s>
<s> Ho è </s>
<s> Ho Ho </s>
<s> Ho Ho </s>
<s> Ho Ho </s>
<s> Ho Ho </s>
Epoch 3, Update 100, Cost = 24,2839
Epoch 6, Update 200, Cost = 16,5616
Epoch 9, Update 300, Cost = 8,5696
Translations:
<s> Buongiorno </s>
<s> Sono la </s>
<s> Sono la </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Questa è la casa </s>
<s> Il gatto dorme </s>
<s> Sono lo </s>
<s> Ho andare </s>
<s> Ho lo </s>
Epoch 3, Update 100, Cost = 22,6819
Epoch 6, Update 200, Cost = 12,2692
Epoch 9, Update 300, Cost = 5,0281
Epoch 12, Update 400, Cost = 2,7495
Epoch 15, Update 500, Cost = 2,4250
Epoch 18, Update 600, Cost = 1,5501
Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Il gatto dorme </s>
<s> Il cane abbaia </s>
<s> Lui gioca a </s>
<s> Vado a scuola </s>
<s> Prendo il treno </s>
           */

            /*
            Translations:
<s> Mi </s>
<s> Mi un </s>
<s> Ho Ho </s>
<s> È È </s>
<s> Dove andiamo ? </s>
<s> Voglio andare a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a
<s> Il Il </s>
<s> Sto un un un </s>
<s> Sto un </s>
<s> Prendo un </s>
Epoch 3, Update 100, Cost = 9,0532
Epoch 6, Update 200, Cost = 12,7345
Epoch 9, Update 300, Cost = 3,4307
Translations:
<s> Ho all'università </s>
<s> Voglio la </s>
<s> Ho andare </s>
<s> Il </s>
<s> Dove anni ? </s>
<s> Voglio andare a </s>
<s> Il all'università </s>
<s> Ho andare </s>
<s> Ho andare </s>
<s> Prendo all'università </s>
Epoch 3, Update 100, Cost = 21,6966
Epoch 6, Update 200, Cost = 22,6786
Epoch 9, Update 300, Cost = 4,2734
Epoch 12, Update 400, Cost = 2,3875
Epoch 15, Update 500, Cost = 4,4991
Epoch 18, Update 600, Cost = 2,1510
Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Dove pronto ? </s>
<s> Voglio andare a casa </s>
<s> Il gatto dorme </s>
<s> Lui gioca a </s>
<s> Facciamo una passeggiata </s>
<s> Prendo il treno </s>
Epoch 3, Update 100, Cost = 8,3000
Epoch 6, Update 200, Cost = 7,5866
Epoch 9, Update 300, Cost = 9,8998
Epoch 12, Update 400, Cost = 5,7521
Epoch 15, Update 500, Cost = 4,5178
Epoch 18, Update 600, Cost = 2,9361
Epoch 21, Update 700, Cost = 3,9452
Epoch 24, Update 800, Cost = 3,4020
Epoch 27, Update 900, Cost = 3,2601
Epoch 30, Update 1000, Cost = 2,7955
Epoch 33, Update 1100, Cost = 2,8643
Epoch 36, Update 1200, Cost = 2,8305
Epoch 39, Update 1300, Cost = 2,5728
Epoch 42, Update 1400, Cost = 2,6351
Epoch 45, Update 1500, Cost = 2,4543
Epoch 48, Update 1600, Cost = 2,4435
Epoch 51, Update 1700, Cost = 2,3986
Epoch 54, Update 1800, Cost = 2,2639
Translations:
<s> Buongiorno </s>
<s> Ti amo </s>
<s> Sono cucinando </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Lui è la casa </s>
<s> Il gatto dorme </s>
<s> Sono andare a </s>
<s> Facciamo una passeggiata </s>
<s> Ti amo </s>
Epoch 3, Update 100, Cost = 16,8281
Epoch 6, Update 200, Cost = 7,4273
Epoch 9, Update 300, Cost = 14,2732
Epoch 12, Update 400, Cost = 8,1511
Epoch 15, Update 500, Cost = 8,9696
Epoch 18, Update 600, Cost = 3,7951
Epoch 21, Update 700, Cost = 2,3505
Epoch 24, Update 800, Cost = 1,5570
Epoch 27, Update 900, Cost = 1,1487
Epoch 30, Update 1000, Cost = 0,9536
Epoch 33, Update 1100, Cost = 0,8058
Epoch 36, Update 1200, Cost = 0,6937
Epoch 39, Update 1300, Cost = 0,6215
Epoch 42, Update 1400, Cost = 0,5624
Epoch 45, Update 1500, Cost = 0,5246
Epoch 48, Update 1600, Cost = 0,5018
Epoch 51, Update 1700, Cost = 0,4687
Epoch 54, Update 1800, Cost = 0,4416
Epoch 57, Update 1900, Cost = 0,4179
Epoch 60, Update 2000, Cost = 0,4047
Epoch 63, Update 2100, Cost = 0,3871
Epoch 66, Update 2200, Cost = 0,3953
Translations:
<s> Buongiorno </s>
<s> Ti amo </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Il gatto dorme </s>
<s> Il cane abbaia </s>
<s> Suono la chitarra </s>
<s> Voglio la chitarra </s>
<s> Prendo il treno </s>
Epoch 3, Update 100, Cost = 14,2739
Epoch 6, Update 200, Cost = 10,4763
Epoch 9, Update 300, Cost = 1,7511
Epoch 12, Update 400, Cost = 5,2621
Epoch 15, Update 500, Cost = 2,9768
Epoch 18, Update 600, Cost = 1,8248
Epoch 21, Update 700, Cost = 1,0250
Epoch 24, Update 800, Cost = 0,6518
Epoch 27, Update 900, Cost = 0,4339
Epoch 30, Update 1000, Cost = 0,3284
Epoch 33, Update 1100, Cost = 0,2644
Epoch 36, Update 1200, Cost = 0,2225
Epoch 39, Update 1300, Cost = 0,1924
Epoch 42, Update 1400, Cost = 0,1716
Epoch 45, Update 1500, Cost = 0,1635
Epoch 48, Update 1600, Cost = 0,1573
Epoch 51, Update 1700, Cost = 0,1471
Epoch 54, Update 1800, Cost = 0,1389
Epoch 57, Update 1900, Cost = 0,1321
Epoch 60, Update 2000, Cost = 0,1293
Epoch 63, Update 2100, Cost = 0,1243
Epoch 66, Update 2200, Cost = 0,1232
Epoch 69, Update 2300, Cost = 0,1201
Epoch 72, Update 2400, Cost = 0,1185
Epoch 75, Update 2500, Cost = 0,1172
Epoch 78, Update 2600, Cost = 0,1193
Epoch 81, Update 2700, Cost = 0,1191
Epoch 84, Update 2800, Cost = 0,1164
Epoch 87, Update 2900, Cost = 0,1157
Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Voglio andare a casa </s>
<s> Il gatto dorme </s>
<s> Ho andare buon </s>
<s> Facciamo una </s>
<s> Prendo il treno </s>
             */

            /*
             Translations:
<s> Il </s>
<s> Sto lo </s>
<s> Ho </s>
<s> È un </s>
<s> Cosa ? ? ? </s>
<s> Voglio a </s>
<s> Il un </s>
<s> Sto lo </s>
<s> Sto </s>
<s> Prendo il il </s>
Epoch 3, Update 100, Cost = 11,0024
Epoch 6, Update 200, Cost = 12,2567
Epoch 9, Update 300, Cost = 4,8698
Translations:
<s> Il </s>
<s> Ho piace </s>
<s> Ho andare </s>
<s> È è mia </s>
<s> Dove vivi ? </s>
<s> Che è il </s>
<s> Il è ? </s>
<s> Ho andare sì </s>
<s> Ho andare </s>
<s> Ho piace anni </s>
Epoch 3, Update 100, Cost = 20,7829
Epoch 6, Update 200, Cost = 19,5703
Epoch 9, Update 300, Cost = 3,4843
Epoch 12, Update 400, Cost = 2,2084
Epoch 15, Update 500, Cost = 6,5484
Epoch 18, Update 600, Cost = 3,3870
Translations:
<s> Buongiorno </s>
<s> Amo piace musica </s>
<s> Ho paura </s>
<s> È pericoloso </s>
<s> Dove vivi ? </s>
<s> Il gatto dorme </s>
<s> Il cane abbaia </s>
<s> Suono la chitarra </s>
<s> Suono andare </s>
<s> Amo piace viaggiare </s>
Epoch 3, Update 100, Cost = 14,6288
Epoch 6, Update 200, Cost = 5,2641
Epoch 9, Update 300, Cost = 8,0051
Epoch 12, Update 400, Cost = 5,0651
Epoch 15, Update 500, Cost = 4,5595
Epoch 18, Update 600, Cost = 3,0583
Epoch 21, Update 700, Cost = 3,6385
Epoch 24, Update 800, Cost = 2,1855
Epoch 27, Update 900, Cost = 1,7471
Epoch 30, Update 1000, Cost = 1,4707
Epoch 33, Update 1100, Cost = 1,2894
Epoch 36, Update 1200, Cost = 1,1472
Epoch 39, Update 1300, Cost = 1,0411
Epoch 42, Update 1400, Cost = 0,9481
Epoch 45, Update 1500, Cost = 0,9108
Epoch 48, Update 1600, Cost = 0,8844
Epoch 51, Update 1700, Cost = 0,8269
Epoch 54, Update 1800, Cost = 0,7804
Translations:
<s> Buongiorno </s>
<s> Ti amo </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Voglio andare a casa </s>
<s> Il gatto dorme </s>
<s> Non lo </s>
<s> Ho a </s>
<s> Prendo il treno </s>
Epoch 3, Update 100, Cost = 17,4131
Epoch 6, Update 200, Cost = 6,0694
Epoch 9, Update 300, Cost = 19,0388
Epoch 12, Update 400, Cost = 16,0460
Epoch 15, Update 500, Cost = 12,7344
Epoch 18, Update 600, Cost = 10,3988
Epoch 21, Update 700, Cost = 8,8767
Epoch 24, Update 800, Cost = 8,0777
Epoch 27, Update 900, Cost = 7,1642
Epoch 30, Update 1000, Cost = 6,2625
Epoch 33, Update 1100, Cost = 6,2489
Epoch 36, Update 1200, Cost = 6,1176
Epoch 39, Update 1300, Cost = 6,3372
Epoch 42, Update 1400, Cost = 6,2595
Epoch 45, Update 1500, Cost = 6,1839
Epoch 48, Update 1600, Cost = 6,0782
Epoch 51, Update 1700, Cost = 5,9857
Epoch 54, Update 1800, Cost = 6,0240
Epoch 57, Update 1900, Cost = 5,9499
Epoch 60, Update 2000, Cost = 5,9415
Epoch 63, Update 2100, Cost = 5,9636
Epoch 66, Update 2200, Cost = 5,8721
Translations:
<s> Il </s>
<s> Ho amo </s>
<s> Sono </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Ciao al parco </s>
<s> Il un film </s>
<s> Ho andare </s>
<s> Ho andare </s>
<s> Ho amo </s>
Epoch 3, Update 100, Cost = 12,7609
Epoch 6, Update 200, Cost = 7,4277
Epoch 9, Update 300, Cost = 1,7009
Epoch 12, Update 400, Cost = 6,4843
Epoch 15, Update 500, Cost = 2,8455
Epoch 18, Update 600, Cost = 3,0400
Epoch 21, Update 700, Cost = 0,9563
Epoch 24, Update 800, Cost = 0,6398
Epoch 27, Update 900, Cost = 0,4376
Epoch 30, Update 1000, Cost = 0,3230
Epoch 33, Update 1100, Cost = 0,2606
Epoch 36, Update 1200, Cost = 0,2174
Epoch 39, Update 1300, Cost = 0,1879
Epoch 42, Update 1400, Cost = 0,1687
Epoch 45, Update 1500, Cost = 0,1608
Epoch 48, Update 1600, Cost = 0,1539
Epoch 51, Update 1700, Cost = 0,1442
Epoch 54, Update 1800, Cost = 0,1366
Epoch 57, Update 1900, Cost = 0,1298
Epoch 60, Update 2000, Cost = 0,1278
Epoch 63, Update 2100, Cost = 0,1234
Epoch 66, Update 2200, Cost = 0,1214
Epoch 69, Update 2300, Cost = 0,1221
Epoch 72, Update 2400, Cost = 0,1207
Epoch 75, Update 2500, Cost = 0,1198
Epoch 78, Update 2600, Cost = 0,1235
Epoch 81, Update 2700, Cost = 0,1226
Epoch 84, Update 2800, Cost = 0,1199
Epoch 87, Update 2900, Cost = 0,1197
Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Voglio andare a casa </s>
<s> Il cane abbaia </s>
<s> Suono andare </s>
<s> Voglio del </s>
<s> Prendo il treno </s>
Epoch 3, Update 100, Cost = 12,3147
Epoch 6, Update 200, Cost = 4,5305
Epoch 9, Update 300, Cost = 7,8595
Epoch 12, Update 400, Cost = 3,6536
Epoch 15, Update 500, Cost = 3,2859
Epoch 18, Update 600, Cost = 1,3627
Epoch 21, Update 700, Cost = 1,0209
Epoch 24, Update 800, Cost = 0,6385
Epoch 27, Update 900, Cost = 0,4525
Epoch 30, Update 1000, Cost = 0,3578
Epoch 33, Update 1100, Cost = 0,2949
Epoch 36, Update 1200, Cost = 0,2473
Epoch 39, Update 1300, Cost = 0,2157
Epoch 42, Update 1400, Cost = 0,1930
Epoch 45, Update 1500, Cost = 0,1793
Epoch 48, Update 1600, Cost = 0,1732
Epoch 51, Update 1700, Cost = 0,1608
Epoch 54, Update 1800, Cost = 0,1514
Epoch 57, Update 1900, Cost = 0,1429
Epoch 60, Update 2000, Cost = 0,1382
Epoch 63, Update 2100, Cost = 0,1326
Epoch 66, Update 2200, Cost = 0,1304
Epoch 69, Update 2300, Cost = 0,1271
Epoch 72, Update 2400, Cost = 0,1254
Epoch 75, Update 2500, Cost = 0,1242
Epoch 78, Update 2600, Cost = 0,1241
Epoch 81, Update 2700, Cost = 0,1228
Epoch 84, Update 2800, Cost = 0,1194
Epoch 87, Update 2900, Cost = 0,1188
Epoch 90, Update 3000, Cost = 0,1175
Epoch 93, Update 3100, Cost = 0,1146
Epoch 96, Update 3200, Cost = 0,1123
Epoch 99, Update 3300, Cost = 0,1098
Epoch 103, Update 3400, Cost = 0,5908
Epoch 106, Update 3500, Cost = 0,3341
Epoch 109, Update 3600, Cost = 0,2476
Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Voglio andare a casa </s>
<s> Il gatto dorme </s>
<s> Lui gioca a </s>
<s> Facciamo una passeggiata </s>
<s> Prendo il treno </s>
             */

            /*
             Translations:
<s> Il </s>
<s> Ho </s>
<s> Ho </s>
<s> È è </s>
<s> Cosa ? ? ? ? </s>
<s> Il Il </s>
<s> Il Il </s>
<s> Ho lo </s>
<s> Il lo </s>
<s> Ho lo </s>
Epoch 3, Update 100, Cost = 11,3111
Epoch 6, Update 200, Cost = 12,5751
Epoch 9, Update 300, Cost = 3,1330
Translations:
<s> Il </s>
<s> Ho andare </s>
<s> Ho andare </s>
<s> È è </s>
<s> Dove ripetere ? </s>
<s> Il una </s>
<s> Il cane </s>
<s> Ho andare </s>
<s> Ho andare </s>
<s> Non venti </s>
Epoch 3, Update 100, Cost = 13,5376
Epoch 6, Update 200, Cost = 19,1349
Epoch 9, Update 300, Cost = 4,1342
Epoch 12, Update 400, Cost = 2,4389
Epoch 15, Update 500, Cost = 7,8394
Epoch 18, Update 600, Cost = 3,4913
Translations:
<s> Buongiorno </s>
<s> Amo la musica musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Cosa anni ? </s>
<s> Voglio andare andare a casa casa </s>
<s> Il Il dorme dorme </s>
<s> Suono piace a </s>
<s> Facciamo una passeggiata </s>
<s> Prendo il treno </s>
Epoch 3, Update 100, Cost = 10,1668
Epoch 6, Update 200, Cost = 6,0692
Epoch 9, Update 300, Cost = 11,1293
Epoch 12, Update 400, Cost = 6,8102
Epoch 15, Update 500, Cost = 4,7567
Epoch 18, Update 600, Cost = 2,8944
Epoch 21, Update 700, Cost = 3,0382
Epoch 24, Update 800, Cost = 1,9322
Epoch 27, Update 900, Cost = 1,5109
Epoch 30, Update 1000, Cost = 1,3053
Epoch 33, Update 1100, Cost = 1,1937
Epoch 36, Update 1200, Cost = 1,1048
Epoch 39, Update 1300, Cost = 1,0842
Epoch 42, Update 1400, Cost = 1,0103
Epoch 45, Update 1500, Cost = 0,9511
Epoch 48, Update 1600, Cost = 0,9109
Epoch 51, Update 1700, Cost = 0,8688
Epoch 54, Update 1800, Cost = 0,8319
Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Il sole splende </s>
<s> Il cane abbaia </s>
<s> Ho la chitarra </s>
<s> Prendo il treno </s>
<s> Amo la viaggiare </s>
Epoch 3, Update 100, Cost = 14,6711
Epoch 6, Update 200, Cost = 4,9375
Epoch 9, Update 300, Cost = 14,5371
Epoch 12, Update 400, Cost = 6,9019
Epoch 15, Update 500, Cost = 6,2199
Epoch 18, Update 600, Cost = 3,1969
Epoch 21, Update 700, Cost = 2,1399
Epoch 24, Update 800, Cost = 1,4890
Epoch 27, Update 900, Cost = 1,1993
Epoch 30, Update 1000, Cost = 1,0212
Epoch 33, Update 1100, Cost = 0,8657
Epoch 36, Update 1200, Cost = 0,7579
Epoch 39, Update 1300, Cost = 0,6612
Epoch 42, Update 1400, Cost = 0,5928
Epoch 45, Update 1500, Cost = 0,5771
Epoch 48, Update 1600, Cost = 0,5472
Epoch 51, Update 1700, Cost = 0,5121
Epoch 54, Update 1800, Cost = 0,4850
Epoch 57, Update 1900, Cost = 0,4777
Epoch 60, Update 2000, Cost = 0,4815
Epoch 63, Update 2100, Cost = 0,4604
Epoch 66, Update 2200, Cost = 0,4506
Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Cosa vivi ? </s>
<s> Voglio andare a casa </s>
<s> Il gatto dorme </s>
<s> Lui gioca a </s>
<s> Facciamo una </s>
<s> Prendo il treno </s>
Epoch 3, Update 100, Cost = 20,1165
Epoch 6, Update 200, Cost = 8,0200
Epoch 9, Update 300, Cost = 2,2912
Epoch 12, Update 400, Cost = 10,9793
Epoch 15, Update 500, Cost = 6,3909
Epoch 18, Update 600, Cost = 4,2307
Epoch 21, Update 700, Cost = 2,6597
Epoch 24, Update 800, Cost = 1,7938
Epoch 27, Update 900, Cost = 1,4230
Epoch 30, Update 1000, Cost = 1,1566
Epoch 33, Update 1100, Cost = 1,0268
Epoch 36, Update 1200, Cost = 0,8712
Epoch 39, Update 1300, Cost = 0,7501
Epoch 42, Update 1400, Cost = 0,6719
Epoch 45, Update 1500, Cost = 0,6458
Epoch 48, Update 1600, Cost = 0,6282
Epoch 51, Update 1700, Cost = 0,5877
Epoch 54, Update 1800, Cost = 0,5549
Epoch 57, Update 1900, Cost = 0,5283
Epoch 60, Update 2000, Cost = 0,5199
Epoch 63, Update 2100, Cost = 0,4982
Epoch 66, Update 2200, Cost = 0,5212
Epoch 69, Update 2300, Cost = 0,5053
Epoch 72, Update 2400, Cost = 0,4952
Epoch 75, Update 2500, Cost = 0,4881
Epoch 78, Update 2600, Cost = 0,4968
Epoch 81, Update 2700, Cost = 0,4965
Epoch 84, Update 2800, Cost = 0,4846
Epoch 87, Update 2900, Cost = 0,4841
Translations:
<s> Buongiorno </s>
<s> Amo la musica </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Il gatto dorme </s>
<s> Il cane abbaia </s>
<s> Sono triste </s>
<s> Facciamo una </s>
<s> Prendo il treno </s>
Epoch 3, Update 100, Cost = 11,4511
Epoch 6, Update 200, Cost = 5,4574
Epoch 9, Update 300, Cost = 10,8205
Epoch 12, Update 400, Cost = 5,3894
Epoch 15, Update 500, Cost = 4,6199
Epoch 18, Update 600, Cost = 2,5564
Epoch 21, Update 700, Cost = 2,1982
Epoch 24, Update 800, Cost = 1,4724
Epoch 27, Update 900, Cost = 1,1853
Epoch 30, Update 1000, Cost = 0,9584
Epoch 33, Update 1100, Cost = 0,8302
Epoch 36, Update 1200, Cost = 0,7289
Epoch 39, Update 1300, Cost = 0,6570
Epoch 42, Update 1400, Cost = 0,6144
Epoch 45, Update 1500, Cost = 0,5927
Epoch 48, Update 1600, Cost = 0,5984
Epoch 51, Update 1700, Cost = 0,5616
Epoch 54, Update 1800, Cost = 0,5309
Epoch 57, Update 1900, Cost = 0,5023
Epoch 60, Update 2000, Cost = 0,4925
Epoch 63, Update 2100, Cost = 0,4735
Epoch 66, Update 2200, Cost = 0,4604
Epoch 69, Update 2300, Cost = 0,5203
Epoch 72, Update 2400, Cost = 0,5052
Epoch 75, Update 2500, Cost = 0,4916
Epoch 78, Update 2600, Cost = 0,4906
Epoch 81, Update 2700, Cost = 0,4811
Epoch 84, Update 2800, Cost = 0,4667
Epoch 87, Update 2900, Cost = 0,4589
Epoch 90, Update 3000, Cost = 0,4491
Epoch 93, Update 3100, Cost = 0,4567
Epoch 96, Update 3200, Cost = 0,4982
Epoch 99, Update 3300, Cost = 0,4852
Epoch 103, Update 3400, Cost = 1,2133
Epoch 106, Update 3500, Cost = 1,4002
Epoch 109, Update 3600, Cost = 1,0259
Translations:
<s> Buongiorno </s>
<s> Ti amo </s>
<s> Sono malato </s>
<s> È bello </s>
<s> Dove vivi ? </s>
<s> Voglio andare a casa </s>
<s> Il gatto dorme </s>
<s> Suono la chitarra </s>
<s> Vado a scuola </s>
<s> Prendo il treno </s>
             */

            Console.ReadLine();
        }
    }
}
