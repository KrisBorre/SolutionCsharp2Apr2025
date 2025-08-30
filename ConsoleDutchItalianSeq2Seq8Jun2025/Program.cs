using Seq2SeqSharp;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Enums;
using Seq2SeqSharp.LearningRate;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;

// ChatGPT, aangepast
namespace ConsoleDutchItalianSeq2Seq8Jun2025
{
    internal class Program
    {
        static void Main(string[] args)
        {
            string srcLang = "NL";
            string tgtLang = "IT";

            // Create in-memory tokenized sentence pairs
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
                ("De kinderen spelen buiten", "I bambini giocano fuori"),
                ("Het is tijd om te slapen", "È ora di dormire"),
                ("Mag ik binnenkomen ?", "Posso entrare ?"),
                ("De deur is open", "La porta è aperta"),
                ("De winkel is gesloten", "Il negozio è chiuso"),
                ("Ik koop brood", "Compro del pane"),
                ("Hoeveel kost het ?", "Quanto costa ?"),
                ("Het is te duur", "È troppo caro"),
                ("Heb je wisselgeld ?", "Hai del resto ?"),
                ("Ik wil graag betalen", "Vorrei pagare"),
                ("Ik wil reserveren", "Vorrei prenotare"),
                ("Voor hoeveel personen ?", "Per quante persone ?"),
                ("Ik ben allergisch", "Sono allergico"),
                ("Wat raad je aan ?", "Cosa consigli ?"),
                ("Het smaakt goed", "È buono"),
                ("Ik ben vol", "Sono sazio"),
                ("Het was lekker", "Era delizioso"),
                ("Tot de volgende keer", "Alla prossima"),
                ("Ik wens je een fijne dag", "Ti auguro una buona giornata"),
                ("Prettige reis", "Buon viaggio"),
                ("Ik ben verloren", "Mi sono perso"),
                ("Ik zoek het hotel", "Sto cercando l'hotel"),
                ("Help me alstublieft", "Aiutami per favore"),
                ("Ik begrijp je", "Ti capisco"),
                ("Ik mis je", "Mi manchi"),
                ("Tot snel", "A presto"),
                ("Ik zie je later", "Ci vediamo dopo"),
                ("Blijf kalm", "Rimani calmo"),
                ("Veel geluk", "Buona fortuna"),
                ("Gefeliciteerd", "Congratulazioni")
            };


            string srcTrainFile = "train.nl.snt";
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
                MaxEpochNum = 50, // 40, // 30, // 20,
                MaxTokenSizePerBatch = 10,
                ValMaxTokenSizePerBatch = 10,
                MaxSrcSentLength = 20,
                MaxTgtSentLength = 20,
                PaddingType = PaddingEnums.AllowPadding,
                TooLongSequence = TooLongSequence.Ignore,
                ProcessorType = ProcessorTypeEnums.CPU,
                ModelFilePath = "nl2it_epochs50.model",
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
            Console.WriteLine($"Vocabulary sizes: src = {srcVocab.Count}, tgt = {tgtVocab.Count}");

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
            opts.ModelFilePath = "nl2it_epochs50.model.trained";
            var inferModel = new Seq2Seq(opts);

            string testInputPath = "test_input.nl.snt";
            string testOutputPath = "test_output.it.snt";
            File.WriteAllLines(testInputPath, new[]
            {
                "Hoe laat is het ?",
                "Dit is mijn huis",
                "Ik hou van mijn lerares en mijn boek"
            });

            inferModel.Test(
                inputTestFile: testInputPath,
                outputFile: testOutputPath,
                batchSize: 1,
                decodingOptions: opts.CreateDecodingOptions(), null, null);

            Console.WriteLine("\nTranslations:");
            foreach (var line in File.ReadLines(testOutputPath))
            {
                Console.WriteLine(line);
            }

            #region 20 epochs

            /*    
     Vocabulary sizes: src = 196, tgt = 210
Epoch 1, Update 100, Cost = 16,8940
Epoch 3, Update 200, Cost = 12,4315
Epoch 5, Update 300, Cost = 7,1162
Epoch 7, Update 400, Cost = 4,0021
Epoch 9, Update 500, Cost = 2,2549
Epoch 11, Update 600, Cost = 1,5738
Epoch 13, Update 700, Cost = 1,0660
Epoch 15, Update 800, Cost = 0,8347
Epoch 17, Update 900, Cost = 0,3797
Epoch 19, Update 1000, Cost = 0,3006

Translations:
<s> Che ore sono ? </s>
<s> Questa è la mia casa </s>
<s> Questa è mia </s>
             */

            /*
             Vocabulary sizes: src = 196, tgt = 210
Epoch 1, Update 100, Cost = 17,7497
Epoch 3, Update 200, Cost = 15,3366
Epoch 5, Update 300, Cost = 12,9237
Epoch 7, Update 400, Cost = 8,5636
Epoch 9, Update 500, Cost = 6,5001
Epoch 11, Update 600, Cost = 4,2308
Epoch 13, Update 700, Cost = 3,4060
Epoch 15, Update 800, Cost = 2,6588
Epoch 17, Update 900, Cost = 1,8167
Epoch 19, Update 1000, Cost = 1,6301

Translations:
<s> Che ore sono ? </s>
<s> Questa è la mia casa </s>
<s> Questo è un libro </s>


             */

            /*
             Vocabulary sizes: src = 196, tgt = 210
Epoch 1, Update 100, Cost = 18,0017
Epoch 3, Update 200, Cost = 16,1521
Epoch 5, Update 300, Cost = 14,9932
Epoch 7, Update 400, Cost = 12,7116
Epoch 9, Update 500, Cost = 10,8552
Epoch 11, Update 600, Cost = 9,3219
Epoch 13, Update 700, Cost = 8,2507
Epoch 15, Update 800, Cost = 7,3443
Epoch 17, Update 900, Cost = 6,0745
Epoch 19, Update 1000, Cost = 5,6996

Translations:
<s> Che ore hai ? </s>
<s> Questa è la mia </s>
<s> Ti la di </s>

             */

            /*
             Vocabulary sizes: src = 196, tgt = 210
Epoch 1, Update 100, Cost = 16,1909
Epoch 3, Update 200, Cost = 10,4375
Epoch 5, Update 300, Cost = 7,1947
Epoch 7, Update 400, Cost = 4,3650
Epoch 9, Update 500, Cost = 3,3432
Epoch 11, Update 600, Cost = 2,1711
Epoch 13, Update 700, Cost = 1,5518
Epoch 15, Update 800, Cost = 1,2634
Epoch 17, Update 900, Cost = 0,7628
Epoch 19, Update 1000, Cost = 0,6802

Translations:
<s> Che ore sono ? </s>
<s> Questa è la mia casa </s>
<s> Sto è un libro </s>


             */

            #endregion

            #region 30 epochs

            /*
             Vocabulary sizes: src = 196, tgt = 210
Epoch 1, Update 100, Cost = 17,9237
Epoch 3, Update 200, Cost = 16,1788
Epoch 5, Update 300, Cost = 15,6216
Epoch 7, Update 400, Cost = 14,5854
Epoch 9, Update 500, Cost = 12,5410
Epoch 11, Update 600, Cost = 11,2457
Epoch 13, Update 700, Cost = 9,8591
Epoch 15, Update 800, Cost = 8,9291
Epoch 17, Update 900, Cost = 7,5386
Epoch 19, Update 1000, Cost = 7,2009
Epoch 21, Update 1100, Cost = 6,9109
Epoch 23, Update 1200, Cost = 6,8178
Epoch 25, Update 1300, Cost = 6,6257
Epoch 27, Update 1400, Cost = 6,5245
Epoch 29, Update 1500, Cost = 6,3186

Translations:
<s> Che ore hai ? </s>
<s> Questa è è mia </s>
<s> Ti la treno </s>
             */

            /*
             Vocabulary sizes: src = 196, tgt = 210
Epoch 1, Update 100, Cost = 16,4513
Epoch 3, Update 200, Cost = 11,7801
Epoch 5, Update 300, Cost = 9,1080
Epoch 7, Update 400, Cost = 5,6337
Epoch 9, Update 500, Cost = 6,5781
Epoch 11, Update 600, Cost = 3,9109
Epoch 13, Update 700, Cost = 3,0841
Epoch 15, Update 800, Cost = 2,8947
Epoch 17, Update 900, Cost = 1,9451
Epoch 19, Update 1000, Cost = 1,7569
Epoch 21, Update 1100, Cost = 1,7307
Epoch 23, Update 1200, Cost = 1,6773
Epoch 25, Update 1300, Cost = 1,6528
Epoch 27, Update 1400, Cost = 1,6595
Epoch 29, Update 1500, Cost = 1,5439

Translations:
<s> Che ore sono ? </s>
<s> Questa è la mia </s>
<s> Ti la musica </s>
             */

            /*
             Vocabulary sizes: src = 196, tgt = 210
Epoch 1, Update 100, Cost = 16,1372
Epoch 3, Update 200, Cost = 10,5784
Epoch 5, Update 300, Cost = 8,6644
Epoch 7, Update 400, Cost = 4,8173
Epoch 9, Update 500, Cost = 4,7915
Epoch 11, Update 600, Cost = 2,7054
Epoch 13, Update 700, Cost = 2,1642
Epoch 15, Update 800, Cost = 1,6663
Epoch 17, Update 900, Cost = 1,0598
Epoch 19, Update 1000, Cost = 0,9598
Epoch 21, Update 1100, Cost = 0,9054
Epoch 23, Update 1200, Cost = 0,8633
Epoch 25, Update 1300, Cost = 0,8283
Epoch 27, Update 1400, Cost = 0,8365
Epoch 29, Update 1500, Cost = 0,8457

Translations:
<s> Che ore sono ? </s>
<s> Questa è la mia casa </s>
<s> Lui è la mia </s>


             */

            /*
             Vocabulary sizes: src = 196, tgt = 210
Epoch 1, Update 100, Cost = 16,1556
Epoch 3, Update 200, Cost = 9,7675
Epoch 5, Update 300, Cost = 7,2183
Epoch 7, Update 400, Cost = 4,5447
Epoch 9, Update 500, Cost = 4,0661
Epoch 11, Update 600, Cost = 2,5963
Epoch 13, Update 700, Cost = 1,9661
Epoch 15, Update 800, Cost = 1,6466
Epoch 17, Update 900, Cost = 1,1042
Epoch 19, Update 1000, Cost = 0,9420
Epoch 21, Update 1100, Cost = 0,9004
Epoch 23, Update 1200, Cost = 0,8713
Epoch 25, Update 1300, Cost = 0,8478
Epoch 27, Update 1400, Cost = 0,8652
Epoch 29, Update 1500, Cost = 0,8034

Translations:
<s> Che ore sono ? </s>
<s> Questa è la mia casa </s>
<s> Questo è un libro </s>
             */

            #endregion

            #region 40 epochs

            /*
             Vocabulary sizes: src = 196, tgt = 210
Epoch 1, Update 100, Cost = 17,2198
Epoch 3, Update 200, Cost = 13,8847
Epoch 5, Update 300, Cost = 10,2816
Epoch 7, Update 400, Cost = 6,3374
Epoch 9, Update 500, Cost = 5,6401
Epoch 11, Update 600, Cost = 4,0781
Epoch 13, Update 700, Cost = 3,5601
Epoch 15, Update 800, Cost = 3,0151
Epoch 17, Update 900, Cost = 2,0867
Epoch 19, Update 1000, Cost = 1,9625
Epoch 21, Update 1100, Cost = 1,9296
Epoch 23, Update 1200, Cost = 1,8390
Epoch 25, Update 1300, Cost = 1,8733
Epoch 27, Update 1400, Cost = 1,9276
Epoch 29, Update 1500, Cost = 1,8352
Epoch 31, Update 1600, Cost = 1,9678
Epoch 33, Update 1700, Cost = 2,1607
Epoch 35, Update 1800, Cost = 2,2422
Epoch 37, Update 1900, Cost = 2,3161
Epoch 39, Update 2000, Cost = 2,2835

Translations:
<s> Che ore sono ? </s>
<s> Questa è la mia </s>
<s> Ti amo musica </s>
             */

            /*
             Vocabulary sizes: src = 196, tgt = 210
Epoch 1, Update 100, Cost = 17,7296
Epoch 3, Update 200, Cost = 14,7695
Epoch 5, Update 300, Cost = 12,9467
Epoch 7, Update 400, Cost = 9,2200
Epoch 9, Update 500, Cost = 6,9098
Epoch 11, Update 600, Cost = 4,4860
Epoch 13, Update 700, Cost = 3,5278
Epoch 15, Update 800, Cost = 2,6942
Epoch 17, Update 900, Cost = 1,7952
Epoch 19, Update 1000, Cost = 1,5481
Epoch 21, Update 1100, Cost = 1,3414
Epoch 23, Update 1200, Cost = 1,2408
Epoch 25, Update 1300, Cost = 1,1490
Epoch 27, Update 1400, Cost = 1,1416
Epoch 29, Update 1500, Cost = 1,1059
Epoch 31, Update 1600, Cost = 1,1568
Epoch 33, Update 1700, Cost = 1,2370
Epoch 35, Update 1800, Cost = 1,3005
Epoch 37, Update 1900, Cost = 1,4112
Epoch 39, Update 2000, Cost = 1,5802

Translations:
<s> Che ore sono ? </s>
<s> Questa è la mia casa </s>
<s> Questo è un libro </s>

             */

            /*
             Vocabulary sizes: src = 196, tgt = 210
Epoch 1, Update 100, Cost = 17,7066
Epoch 3, Update 200, Cost = 15,4078
Epoch 5, Update 300, Cost = 13,0808
Epoch 7, Update 400, Cost = 8,4095
Epoch 9, Update 500, Cost = 6,0253
Epoch 11, Update 600, Cost = 3,5780
Epoch 13, Update 700, Cost = 2,8452
Epoch 15, Update 800, Cost = 2,0717
Epoch 17, Update 900, Cost = 1,2480
Epoch 19, Update 1000, Cost = 1,0688
Epoch 21, Update 1100, Cost = 0,9250
Epoch 23, Update 1200, Cost = 0,8664
Epoch 25, Update 1300, Cost = 0,8018
Epoch 27, Update 1400, Cost = 0,7836
Epoch 29, Update 1500, Cost = 0,7801
Epoch 31, Update 1600, Cost = 0,8200
Epoch 33, Update 1700, Cost = 0,8534
Epoch 35, Update 1800, Cost = 0,9110
Epoch 37, Update 1900, Cost = 0,9981
Epoch 39, Update 2000, Cost = 1,1050

Translations:
<s> Che ore sono ? </s>
<s> Questa è la mia casa </s>
<s> Questo è un libro </s>
             */

            /*
             Vocabulary sizes: src = 196, tgt = 210
Epoch 1, Update 100, Cost = 17,0673
Epoch 3, Update 200, Cost = 13,0682
Epoch 5, Update 300, Cost = 10,7678
Epoch 7, Update 400, Cost = 8,2571
Epoch 9, Update 500, Cost = 8,2284
Epoch 11, Update 600, Cost = 7,1968
Epoch 13, Update 700, Cost = 6,5561
Epoch 15, Update 800, Cost = 6,0252
Epoch 17, Update 900, Cost = 5,5201
Epoch 19, Update 1000, Cost = 5,2705
Epoch 21, Update 1100, Cost = 5,0889
Epoch 23, Update 1200, Cost = 5,1000
Epoch 25, Update 1300, Cost = 5,0316
Epoch 27, Update 1400, Cost = 4,9428
Epoch 29, Update 1500, Cost = 4,6935
Epoch 31, Update 1600, Cost = 4,6425
Epoch 33, Update 1700, Cost = 5,1353
Epoch 35, Update 1800, Cost = 5,2794
Epoch 37, Update 1900, Cost = 5,4723
Epoch 39, Update 2000, Cost = 5,2775

Translations:
<s> Che ore sono ? </s>
<s> Questa è la mia </s>
<s> Sono la </s>
             */

            #endregion

            /*
             Vocabulary sizes: src = 196, tgt = 210
Epoch 1, Update 100, Cost = 17,8016
Epoch 3, Update 200, Cost = 15,8969
Epoch 5, Update 300, Cost = 15,5093
Epoch 7, Update 400, Cost = 14,5868
Epoch 9, Update 500, Cost = 13,4731
Epoch 11, Update 600, Cost = 12,5380
Epoch 13, Update 700, Cost = 11,7291
Epoch 15, Update 800, Cost = 10,9585
Epoch 17, Update 900, Cost = 9,6558
Epoch 19, Update 1000, Cost = 9,4554
Epoch 21, Update 1100, Cost = 9,2127
Epoch 23, Update 1200, Cost = 8,9408
Epoch 25, Update 1300, Cost = 8,7147
Epoch 27, Update 1400, Cost = 8,4448
Epoch 29, Update 1500, Cost = 8,2805
Epoch 31, Update 1600, Cost = 8,2883
Epoch 33, Update 1700, Cost = 8,5022
Epoch 35, Update 1800, Cost = 8,8335
Epoch 37, Update 1900, Cost = 9,2340
Epoch 39, Update 2000, Cost = 9,0063
Epoch 41, Update 2100, Cost = 8,8793
Epoch 43, Update 2200, Cost = 9,2164
Epoch 45, Update 2300, Cost = 9,7104
Epoch 47, Update 2400, Cost = 9,6827
Epoch 49, Update 2500, Cost = 14,8387

Translations:
<s> Dove è ? ? </s>
<s> Questo è è mia </s>
<s> Sono la </s>
             */

            /*
             Vocabulary sizes: src = 196, tgt = 210
Epoch 1, Update 100, Cost = 16,9802
Epoch 3, Update 200, Cost = 12,6852
Epoch 5, Update 300, Cost = 9,0431
Epoch 7, Update 400, Cost = 5,1712
Epoch 9, Update 500, Cost = 4,1132
Epoch 11, Update 600, Cost = 2,7419
Epoch 13, Update 700, Cost = 2,2039
Epoch 15, Update 800, Cost = 1,7430
Epoch 17, Update 900, Cost = 1,1055
Epoch 19, Update 1000, Cost = 0,9998
Epoch 21, Update 1100, Cost = 0,9183
Epoch 23, Update 1200, Cost = 0,8674
Epoch 25, Update 1300, Cost = 0,8292
Epoch 27, Update 1400, Cost = 0,8371
Epoch 29, Update 1500, Cost = 0,8345
Epoch 31, Update 1600, Cost = 0,8604
Epoch 33, Update 1700, Cost = 0,9325
Epoch 35, Update 1800, Cost = 0,9886
Epoch 37, Update 1900, Cost = 1,0537
Epoch 39, Update 2000, Cost = 1,1908
Epoch 41, Update 2100, Cost = 1,3014
Epoch 43, Update 2200, Cost = 1,5955
Epoch 45, Update 2300, Cost = 1,9357
Epoch 47, Update 2400, Cost = 1,5120
Epoch 49, Update 2500, Cost = 3,5112

Translations:
<s> Che ore sono ? </s>
<s> Questa è la mia casa </s>
<s> Questo è un libro </s>
             */

            /*
             Vocabulary sizes: src = 196, tgt = 210
Epoch 1, Update 100, Cost = 16,0597
Epoch 3, Update 200, Cost = 9,6784
Epoch 5, Update 300, Cost = 6,9964
Epoch 7, Update 400, Cost = 4,2584
Epoch 9, Update 500, Cost = 3,2466
Epoch 11, Update 600, Cost = 2,2546
Epoch 13, Update 700, Cost = 1,7009
Epoch 15, Update 800, Cost = 1,3785
Epoch 17, Update 900, Cost = 0,8600
Epoch 19, Update 1000, Cost = 0,7835
Epoch 21, Update 1100, Cost = 0,7334
Epoch 23, Update 1200, Cost = 0,7114
Epoch 25, Update 1300, Cost = 0,6920
Epoch 27, Update 1400, Cost = 0,7027
Epoch 29, Update 1500, Cost = 0,7159
Epoch 31, Update 1600, Cost = 0,7377
Epoch 33, Update 1700, Cost = 0,8061
Epoch 35, Update 1800, Cost = 0,8733
Epoch 37, Update 1900, Cost = 0,9299
Epoch 39, Update 2000, Cost = 1,0659
Epoch 41, Update 2100, Cost = 1,0742
Epoch 43, Update 2200, Cost = 1,3247
Epoch 45, Update 2300, Cost = 1,6331
Epoch 47, Update 2400, Cost = 1,4553
Epoch 49, Update 2500, Cost = 3,4368

Translations:
<s> Che ore sono ? </s>
<s> Questa è la mia casa </s>
<s> Lui è la </s>
             */

            Console.ReadLine();
        }
    }
}
