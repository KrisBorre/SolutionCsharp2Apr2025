using Seq2SeqSharp;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Enums;
using Seq2SeqSharp.LearningRate;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;

// ChatGPT, aangepast
namespace ConsoleDutchItalianSeq2Seq7Jun2025
{
    internal class Program
    {
        static void Main(string[] args)
        {
            string srcLang = "NL";
            string tgtLang = "IT";

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
                HiddenSize = 512,
                SrcEmbeddingDim = 512,
                TgtEmbeddingDim = 512,
                MaxEpochNum = 100, // 300,
                MaxTokenSizePerBatch = 10,
                ValMaxTokenSizePerBatch = 10,
                MaxSrcSentLength = 20,
                MaxTgtSentLength = 20,
                PaddingType = PaddingEnums.AllowPadding,
                TooLongSequence = TooLongSequence.Ignore,
                ProcessorType = ProcessorTypeEnums.CPU,
                ModelFilePath = "nl2it_epochs100.model",
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
            opts.ModelFilePath = "nl2it_epochs100.model.trained";
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

            /*    
     
            Vocabulary sizes: src = 196, tgt = 210
Epoch 1, Update 100, Cost = 16,4741
Epoch 3, Update 200, Cost = 11,3094
Epoch 5, Update 300, Cost = 7,0535
Epoch 7, Update 400, Cost = 9,9370
Epoch 9, Update 500, Cost = 6,8422
Epoch 11, Update 600, Cost = 5,4111
Epoch 13, Update 700, Cost = 4,5710
Epoch 15, Update 800, Cost = 4,4394
Epoch 17, Update 900, Cost = 3,4013
Epoch 19, Update 1000, Cost = 3,2697
Epoch 21, Update 1100, Cost = 3,0740
Epoch 23, Update 1200, Cost = 2,9395
Epoch 25, Update 1300, Cost = 2,9067
Epoch 27, Update 1400, Cost = 2,8928
Epoch 29, Update 1500, Cost = 2,7959
Epoch 31, Update 1600, Cost = 2,8974
Epoch 33, Update 1700, Cost = 3,1150
Epoch 35, Update 1800, Cost = 3,2672
Epoch 37, Update 1900, Cost = 3,5464
Epoch 39, Update 2000, Cost = 3,7100
Epoch 41, Update 2100, Cost = 3,8834
Epoch 43, Update 2200, Cost = 4,6223
Epoch 45, Update 2300, Cost = 5,4598
Epoch 47, Update 2400, Cost = 4,6408
Epoch 49, Update 2500, Cost = 6,8567
Epoch 50, Update 2600, Cost = 2,7474
Epoch 52, Update 2700, Cost = 2,8238
Epoch 54, Update 2800, Cost = 2,8837
Epoch 56, Update 2900, Cost = 2,8624
Epoch 58, Update 3000, Cost = 2,9552
Epoch 60, Update 3100, Cost = 3,0227
Epoch 62, Update 3200, Cost = 3,0777
Epoch 64, Update 3300, Cost = 3,0484
Epoch 66, Update 3400, Cost = 2,8479
Epoch 68, Update 3500, Cost = 2,6288
Epoch 70, Update 3600, Cost = 2,7299
Epoch 72, Update 3700, Cost = 2,6566
Epoch 74, Update 3800, Cost = 2,7512
Epoch 76, Update 3900, Cost = 2,7575
Epoch 78, Update 4000, Cost = 2,7652
Epoch 80, Update 4100, Cost = 2,8158
Epoch 82, Update 4200, Cost = 2,9675
Epoch 84, Update 4300, Cost = 3,2357
Epoch 86, Update 4400, Cost = 3,3602
Epoch 88, Update 4500, Cost = 3,6433
Epoch 90, Update 4600, Cost = 3,8605
Epoch 92, Update 4700, Cost = 4,1478
Epoch 94, Update 4800, Cost = 4,9675
Epoch 96, Update 4900, Cost = 4,7206
Epoch 98, Update 5000, Cost = 4,9868
Epoch 99, Update 5100, Cost = 2,7081
Epoch 101, Update 5200, Cost = 2,7821
Epoch 103, Update 5300, Cost = 2,8600
Epoch 105, Update 5400, Cost = 2,8232
Epoch 107, Update 5500, Cost = 2,9024
Epoch 109, Update 5600, Cost = 2,9768
Epoch 111, Update 5700, Cost = 3,0478
Epoch 113, Update 5800, Cost = 3,1080
Epoch 115, Update 5900, Cost = 3,0983
Epoch 117, Update 6000, Cost = 2,5738
Epoch 119, Update 6100, Cost = 2,6650
Epoch 121, Update 6200, Cost = 2,7568
Epoch 123, Update 6300, Cost = 2,7222
Epoch 125, Update 6400, Cost = 2,7501
Epoch 127, Update 6500, Cost = 2,7760
Epoch 129, Update 6600, Cost = 2,7331
Epoch 131, Update 6700, Cost = 2,8512
Epoch 133, Update 6800, Cost = 3,0791
Epoch 135, Update 6900, Cost = 3,2452
Epoch 137, Update 7000, Cost = 3,5297
Epoch 139, Update 7100, Cost = 3,6982
Epoch 141, Update 7200, Cost = 3,8744
Epoch 143, Update 7300, Cost = 4,6169
Epoch 145, Update 7400, Cost = 5,4556
Epoch 147, Update 7500, Cost = 4,6378
Epoch 149, Update 7600, Cost = 6,8549
Epoch 150, Update 7700, Cost = 2,7466
Epoch 152, Update 7800, Cost = 2,8233
Epoch 154, Update 7900, Cost = 2,8834
Epoch 156, Update 8000, Cost = 2,8623
Epoch 158, Update 8100, Cost = 2,9551
Epoch 160, Update 8200, Cost = 3,0226
Epoch 162, Update 8300, Cost = 3,0776
Epoch 164, Update 8400, Cost = 3,0484
Epoch 166, Update 8500, Cost = 2,8479
Epoch 168, Update 8600, Cost = 2,6288
Epoch 170, Update 8700, Cost = 2,7299
Epoch 172, Update 8800, Cost = 2,6566
Epoch 174, Update 8900, Cost = 2,7512
Epoch 176, Update 9000, Cost = 2,7575
Epoch 178, Update 9100, Cost = 2,7652
Epoch 180, Update 9200, Cost = 2,8158
Epoch 182, Update 9300, Cost = 2,9675
Epoch 184, Update 9400, Cost = 3,2357
Epoch 186, Update 9500, Cost = 3,3602
Epoch 188, Update 9600, Cost = 3,6433
Epoch 190, Update 9700, Cost = 3,8605
Epoch 192, Update 9800, Cost = 4,1478
Epoch 194, Update 9900, Cost = 4,9675
Epoch 196, Update 10000, Cost = 4,7206
Epoch 198, Update 10100, Cost = 4,9868
Epoch 199, Update 10200, Cost = 2,7081

Translations:
<s> Che ore sono ? </s>
<s> Questa è mia mia </s>
<s> Ti di di </s>

             */


            /*
             Vocabulary sizes: src = 196, tgt = 210
Epoch 1, Update 100, Cost = 17,2930
Epoch 3, Update 200, Cost = 14,7595
Epoch 5, Update 300, Cost = 11,8090
Epoch 7, Update 400, Cost = 10,0793
Epoch 9, Update 500, Cost = 9,2953
Epoch 11, Update 600, Cost = 8,6190
Epoch 13, Update 700, Cost = 7,7226
Epoch 15, Update 800, Cost = 7,1595
Epoch 17, Update 900, Cost = 6,2365
Epoch 19, Update 1000, Cost = 12,6944
Epoch 21, Update 1100, Cost = 12,2424
Epoch 23, Update 1200, Cost = 11,9562
Epoch 25, Update 1300, Cost = 11,7377
Epoch 27, Update 1400, Cost = 11,4669
Epoch 29, Update 1500, Cost = 11,4831
Epoch 31, Update 1600, Cost = 11,6109
Epoch 33, Update 1700, Cost = 12,1155
Epoch 35, Update 1800, Cost = 12,7120
Epoch 37, Update 1900, Cost = 12,9440
Epoch 39, Update 2000, Cost = 12,7706
Epoch 41, Update 2100, Cost = 12,7334
Epoch 43, Update 2200, Cost = 13,6180
Epoch 45, Update 2300, Cost = 14,4966
Epoch 47, Update 2400, Cost = 13,1655
Epoch 49, Update 2500, Cost = 14,8695
Epoch 50, Update 2600, Cost = 10,0868
Epoch 52, Update 2700, Cost = 10,1928
Epoch 54, Update 2800, Cost = 10,2750
Epoch 56, Update 2900, Cost = 10,2345
Epoch 58, Update 3000, Cost = 10,4907
Epoch 60, Update 3100, Cost = 10,6234
Epoch 62, Update 3200, Cost = 10,7877
Epoch 64, Update 3300, Cost = 10,9348
Epoch 66, Update 3400, Cost = 10,6829
Epoch 68, Update 3500, Cost = 10,5641
Epoch 70, Update 3600, Cost = 10,8694
Epoch 72, Update 3700, Cost = 10,8917
Epoch 74, Update 3800, Cost = 11,0623
Epoch 76, Update 3900, Cost = 11,0034
Epoch 78, Update 4000, Cost = 10,9121
Epoch 80, Update 4100, Cost = 11,0701
Epoch 82, Update 4200, Cost = 11,4328
Epoch 84, Update 4300, Cost = 12,0023
Epoch 86, Update 4400, Cost = 12,4319
Epoch 88, Update 4500, Cost = 12,8806
Epoch 90, Update 4600, Cost = 12,7286
Epoch 92, Update 4700, Cost = 12,9650
Epoch 94, Update 4800, Cost = 13,7934
Epoch 96, Update 4900, Cost = 14,4150
Epoch 98, Update 5000, Cost = 13,1766
Epoch 99, Update 5100, Cost = 9,9773
Epoch 101, Update 5200, Cost = 10,1125
Epoch 103, Update 5300, Cost = 10,2320
Epoch 105, Update 5400, Cost = 10,1305
Epoch 107, Update 5500, Cost = 10,3434
Epoch 109, Update 5600, Cost = 10,5685
Epoch 111, Update 5700, Cost = 10,7680
Epoch 113, Update 5800, Cost = 10,8799
Epoch 115, Update 5900, Cost = 10,9761
Epoch 117, Update 6000, Cost = 10,4523
Epoch 119, Update 6100, Cost = 10,6972
Epoch 121, Update 6200, Cost = 10,8574
Epoch 123, Update 6300, Cost = 10,9840
Epoch 125, Update 6400, Cost = 11,0474
Epoch 127, Update 6500, Cost = 10,9237
Epoch 129, Update 6600, Cost = 10,9854
Epoch 131, Update 6700, Cost = 11,1419
Epoch 133, Update 6800, Cost = 11,6591
Epoch 135, Update 6900, Cost = 12,2563
Epoch 137, Update 7000, Cost = 12,8913
Epoch 139, Update 7100, Cost = 12,7361
Epoch 141, Update 7200, Cost = 12,7079
Epoch 143, Update 7300, Cost = 13,5989
Epoch 145, Update 7400, Cost = 14,4807
Epoch 147, Update 7500, Cost = 13,1489
Epoch 149, Update 7600, Cost = 14,8580
Epoch 150, Update 7700, Cost = 10,0779
Epoch 152, Update 7800, Cost = 10,1843
Epoch 154, Update 7900, Cost = 10,2742
Epoch 156, Update 8000, Cost = 10,2340
Epoch 158, Update 8100, Cost = 10,4903
Epoch 160, Update 8200, Cost = 10,6231
Epoch 162, Update 8300, Cost = 10,7874
Epoch 164, Update 8400, Cost = 10,9346
Epoch 166, Update 8500, Cost = 10,6827
Epoch 168, Update 8600, Cost = 10,5638
Epoch 170, Update 8700, Cost = 10,8692
Epoch 172, Update 8800, Cost = 10,8917
Epoch 174, Update 8900, Cost = 11,0623
Epoch 176, Update 9000, Cost = 11,0034
Epoch 178, Update 9100, Cost = 10,9121
Epoch 180, Update 9200, Cost = 11,0701
Epoch 182, Update 9300, Cost = 11,4328
Epoch 184, Update 9400, Cost = 12,0023
Epoch 186, Update 9500, Cost = 12,4319
Epoch 188, Update 9600, Cost = 12,8806
Epoch 190, Update 9700, Cost = 12,7286
Epoch 192, Update 9800, Cost = 12,9650
Epoch 194, Update 9900, Cost = 13,7934
Epoch 196, Update 10000, Cost = 14,4150
Epoch 198, Update 10100, Cost = 13,1766
Epoch 199, Update 10200, Cost = 9,9773
Epoch 201, Update 10300, Cost = 10,1125
Epoch 203, Update 10400, Cost = 10,2320
Epoch 205, Update 10500, Cost = 10,1305
Epoch 207, Update 10600, Cost = 10,3434
Epoch 209, Update 10700, Cost = 10,5685
Epoch 211, Update 10800, Cost = 10,7680
Epoch 213, Update 10900, Cost = 10,8799
Epoch 215, Update 11000, Cost = 10,9761
Epoch 217, Update 11100, Cost = 10,4523
Epoch 219, Update 11200, Cost = 10,6972
Epoch 221, Update 11300, Cost = 10,8574
Epoch 223, Update 11400, Cost = 10,9840
Epoch 225, Update 11500, Cost = 11,0474
Epoch 227, Update 11600, Cost = 10,9237
Epoch 229, Update 11700, Cost = 10,9854
Epoch 231, Update 11800, Cost = 11,1419
Epoch 233, Update 11900, Cost = 11,6591
Epoch 235, Update 12000, Cost = 12,2563
Epoch 237, Update 12100, Cost = 12,8913
Epoch 239, Update 12200, Cost = 12,7361
Epoch 241, Update 12300, Cost = 12,7079
Epoch 243, Update 12400, Cost = 13,5989
Epoch 245, Update 12500, Cost = 14,4807
Epoch 247, Update 12600, Cost = 13,1489
Epoch 249, Update 12700, Cost = 14,8580
Epoch 250, Update 12800, Cost = 10,0779
Epoch 252, Update 12900, Cost = 10,1843
Epoch 254, Update 13000, Cost = 10,2742
Epoch 256, Update 13100, Cost = 10,2340
Epoch 258, Update 13200, Cost = 10,4903
Epoch 260, Update 13300, Cost = 10,6231
Epoch 262, Update 13400, Cost = 10,7874
Epoch 264, Update 13500, Cost = 10,9346
Epoch 266, Update 13600, Cost = 10,6827
Epoch 268, Update 13700, Cost = 10,5638
Epoch 270, Update 13800, Cost = 10,8692
Epoch 272, Update 13900, Cost = 10,8917
Epoch 274, Update 14000, Cost = 11,0623
Epoch 276, Update 14100, Cost = 11,0034
Epoch 278, Update 14200, Cost = 10,9121
Epoch 280, Update 14300, Cost = 11,0701
Epoch 282, Update 14400, Cost = 11,4328
Epoch 284, Update 14500, Cost = 12,0023
Epoch 286, Update 14600, Cost = 12,4319
Epoch 288, Update 14700, Cost = 12,8806
Epoch 290, Update 14800, Cost = 12,7286
Epoch 292, Update 14900, Cost = 12,9650
Epoch 294, Update 15000, Cost = 13,7934
Epoch 296, Update 15100, Cost = 14,4150
Epoch 298, Update 15200, Cost = 13,1766
Epoch 299, Update 15300, Cost = 9,9773

Translations:
<s> Dove è è ? </s>
<s> È è è </s>
<s> Sto la un </s>


             */

            /*
             Vocabulary sizes: src = 196, tgt = 210
Epoch 1, Update 100, Cost = 16,4302
Epoch 3, Update 200, Cost = 10,9017
Epoch 5, Update 300, Cost = 6,4780
Epoch 7, Update 400, Cost = 4,8976
Epoch 9, Update 500, Cost = 3,7101
Epoch 11, Update 600, Cost = 2,8005
Epoch 13, Update 700, Cost = 2,3645
Epoch 15, Update 800, Cost = 2,1088
Epoch 17, Update 900, Cost = 1,4980
Epoch 19, Update 1000, Cost = 7,0526
Epoch 21, Update 1100, Cost = 6,7251
Epoch 23, Update 1200, Cost = 6,5185
Epoch 25, Update 1300, Cost = 6,2622
Epoch 27, Update 1400, Cost = 6,1865
Epoch 29, Update 1500, Cost = 6,3648
Epoch 31, Update 1600, Cost = 6,6397
Epoch 33, Update 1700, Cost = 7,2105
Epoch 35, Update 1800, Cost = 7,7725
Epoch 37, Update 1900, Cost = 7,9399
Epoch 39, Update 2000, Cost = 8,7516
Epoch 41, Update 2100, Cost = 9,4605
Epoch 43, Update 2200, Cost = 11,2362
Epoch 45, Update 2300, Cost = 13,2126
Epoch 47, Update 2400, Cost = 13,4051
Epoch 49, Update 2500, Cost = 24,7338
Epoch 50, Update 2600, Cost = 4,3153
Epoch 52, Update 2700, Cost = 4,4580
Epoch 54, Update 2800, Cost = 4,5827
Epoch 56, Update 2900, Cost = 4,6510
Epoch 58, Update 3000, Cost = 4,8140
Epoch 60, Update 3100, Cost = 4,8431
Epoch 62, Update 3200, Cost = 4,9866
Epoch 64, Update 3300, Cost = 5,1194
Epoch 66, Update 3400, Cost = 5,0473
Epoch 68, Update 3500, Cost = 4,9503
Epoch 70, Update 3600, Cost = 5,2187
Epoch 72, Update 3700, Cost = 5,3822
Epoch 74, Update 3800, Cost = 5,3791
Epoch 76, Update 3900, Cost = 5,4621
Epoch 78, Update 4000, Cost = 5,6530
Epoch 80, Update 4100, Cost = 5,9279
Epoch 82, Update 4200, Cost = 6,3074
Epoch 84, Update 4300, Cost = 6,9287
Epoch 86, Update 4400, Cost = 7,4534
Epoch 88, Update 4500, Cost = 8,3395
Epoch 90, Update 4600, Cost = 9,0247
Epoch 92, Update 4700, Cost = 10,0868
Epoch 94, Update 4800, Cost = 12,0435
Epoch 96, Update 4900, Cost = 13,3904
Epoch 98, Update 5000, Cost = 15,8503
Epoch 99, Update 5100, Cost = 4,2441

Translations:
<s> Dove è ? ? </s>
<s> Il è è casa </s>
<s> Mi la la </s>


             */

            Console.ReadLine();
        }
    }
}
