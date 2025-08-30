using Seq2SeqSharp;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Enums;
using Seq2SeqSharp.LearningRate;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;

// ChatGPT, aangepast
namespace ConsoleDutchItalianSeq2Seq5Jun2025
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
                HiddenSize = 128,
                SrcEmbeddingDim = 128,
                TgtEmbeddingDim = 128,
                MaxEpochNum = 200,
                MaxTokenSizePerBatch = 10,
                ValMaxTokenSizePerBatch = 10,
                MaxSrcSentLength = 20,
                MaxTgtSentLength = 20,
                PaddingType = PaddingEnums.AllowPadding,
                TooLongSequence = TooLongSequence.Ignore,
                ProcessorType = ProcessorTypeEnums.CPU,
                ModelFilePath = "nl2it_epochs200.model",
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
            opts.ModelFilePath = "nl2it_epochs200.model.trained";
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
Epoch 1, Update 100, Cost = 17,3367
Epoch 3, Update 200, Cost = 15,1827
Epoch 5, Update 300, Cost = 13,7287
Epoch 7, Update 400, Cost = 13,2854
Epoch 9, Update 500, Cost = 12,9466
Epoch 11, Update 600, Cost = 12,6726
Epoch 13, Update 700, Cost = 12,4220
Epoch 15, Update 800, Cost = 12,1348
Epoch 17, Update 900, Cost = 11,0391
Epoch 19, Update 1000, Cost = 11,0528
Epoch 21, Update 1100, Cost = 11,1158
Epoch 23, Update 1200, Cost = 11,0536
Epoch 25, Update 1300, Cost = 10,8996
Epoch 27, Update 1400, Cost = 10,6135
Epoch 29, Update 1500, Cost = 10,4686
Epoch 31, Update 1600, Cost = 10,5456
Epoch 33, Update 1700, Cost = 11,0077
Epoch 35, Update 1800, Cost = 11,3072
Epoch 37, Update 1900, Cost = 11,7230
Epoch 39, Update 2000, Cost = 11,9021
Epoch 41, Update 2100, Cost = 11,6147
Epoch 43, Update 2200, Cost = 12,7502
Epoch 45, Update 2300, Cost = 13,5405
Epoch 47, Update 2400, Cost = 12,0856
Epoch 49, Update 2500, Cost = 16,4254
Epoch 50, Update 2600, Cost = 11,9363
Epoch 52, Update 2700, Cost = 12,0430
Epoch 54, Update 2800, Cost = 12,0651
Epoch 56, Update 2900, Cost = 11,8889
Epoch 58, Update 3000, Cost = 12,0700
Epoch 60, Update 3100, Cost = 12,1171
Epoch 62, Update 3200, Cost = 12,0964
Epoch 64, Update 3300, Cost = 11,9471
Epoch 66, Update 3400, Cost = 11,4312
Epoch 68, Update 3500, Cost = 10,9347
Epoch 70, Update 3600, Cost = 11,0814
Epoch 72, Update 3700, Cost = 10,9674
Epoch 74, Update 3800, Cost = 10,9643
Epoch 76, Update 3900, Cost = 10,7392
Epoch 78, Update 4000, Cost = 10,4897
Epoch 80, Update 4100, Cost = 10,5340
Epoch 82, Update 4200, Cost = 10,8030
Epoch 84, Update 4300, Cost = 11,3226
Epoch 86, Update 4400, Cost = 11,4061
Epoch 88, Update 4500, Cost = 11,8147
Epoch 90, Update 4600, Cost = 11,8167
Epoch 92, Update 4700, Cost = 11,9516
Epoch 94, Update 4800, Cost = 12,9525
Epoch 96, Update 4900, Cost = 18,5395
Epoch 98, Update 5000, Cost = 17,2415
Epoch 99, Update 5100, Cost = 13,7908
Epoch 101, Update 5200, Cost = 13,9522
Epoch 103, Update 5300, Cost = 14,1215
Epoch 105, Update 5400, Cost = 13,9490
Epoch 107, Update 5500, Cost = 14,2057
Epoch 109, Update 5600, Cost = 14,3982
Epoch 111, Update 5700, Cost = 14,5338
Epoch 113, Update 5800, Cost = 14,5470
Epoch 115, Update 5900, Cost = 14,4191
Epoch 117, Update 6000, Cost = 13,2305
Epoch 119, Update 6100, Cost = 13,3509
Epoch 121, Update 6200, Cost = 13,5536
Epoch 123, Update 6300, Cost = 13,5930
Epoch 125, Update 6400, Cost = 13,5560
Epoch 127, Update 6500, Cost = 13,3940
Epoch 129, Update 6600, Cost = 13,3060
Epoch 131, Update 6700, Cost = 13,5903
Epoch 133, Update 6800, Cost = 14,2365
Epoch 135, Update 6900, Cost = 14,7050
Epoch 137, Update 7000, Cost = 15,4010
Epoch 139, Update 7100, Cost = 15,6566
Epoch 141, Update 7200, Cost = 15,6293
Epoch 143, Update 7300, Cost = 17,0053
Epoch 145, Update 7400, Cost = 18,6206
Epoch 147, Update 7500, Cost = 17,1719
Epoch 149, Update 7600, Cost = 21,4958
Epoch 150, Update 7700, Cost = 13,8913
Epoch 152, Update 7800, Cost = 14,0703
Epoch 154, Update 7900, Cost = 14,1770
Epoch 156, Update 8000, Cost = 14,0693
Epoch 158, Update 8100, Cost = 14,3326
Epoch 160, Update 8200, Cost = 14,4561
Epoch 162, Update 8300, Cost = 14,5251
Epoch 164, Update 8400, Cost = 14,4343
Epoch 166, Update 8500, Cost = 13,8294
Epoch 168, Update 8600, Cost = 13,2906
Epoch 170, Update 8700, Cost = 13,5372
Epoch 172, Update 8800, Cost = 13,5158
Epoch 174, Update 8900, Cost = 13,5781
Epoch 176, Update 9000, Cost = 13,4867
Epoch 178, Update 9100, Cost = 13,3474
Epoch 180, Update 9200, Cost = 13,4675
Epoch 182, Update 9300, Cost = 13,9213
Epoch 184, Update 9400, Cost = 14,6394
Epoch 186, Update 9500, Cost = 14,8991
Epoch 188, Update 9600, Cost = 15,6237
Epoch 190, Update 9700, Cost = 15,6528
Epoch 192, Update 9800, Cost = 15,9822
Epoch 194, Update 9900, Cost = 17,5214
Epoch 196, Update 10000, Cost = 18,5395
Epoch 198, Update 10100, Cost = 17,2415
Epoch 199, Update 10200, Cost = 13,7908

Translations:
<s> Dove ? ? ? </s>
<s> è è è è </s>
<s> È leggendo </s>

             */


            /*
             Vocabulary sizes: src = 196, tgt = 210
Epoch 1, Update 100, Cost = 17,5876
Epoch 3, Update 200, Cost = 15,7696
Epoch 5, Update 300, Cost = 14,2735
Epoch 7, Update 400, Cost = 13,6828
Epoch 9, Update 500, Cost = 13,2711
Epoch 11, Update 600, Cost = 13,0378
Epoch 13, Update 700, Cost = 12,7961
Epoch 15, Update 800, Cost = 12,4851
Epoch 17, Update 900, Cost = 11,4106
Epoch 19, Update 1000, Cost = 11,4040
Epoch 21, Update 1100, Cost = 11,4606
Epoch 23, Update 1200, Cost = 11,4133
Epoch 25, Update 1300, Cost = 11,2276
Epoch 27, Update 1400, Cost = 10,9109
Epoch 29, Update 1500, Cost = 10,8346
Epoch 31, Update 1600, Cost = 10,9611
Epoch 33, Update 1700, Cost = 11,4271
Epoch 35, Update 1800, Cost = 11,7571
Epoch 37, Update 1900, Cost = 12,1698
Epoch 39, Update 2000, Cost = 12,2290
Epoch 41, Update 2100, Cost = 11,9844
Epoch 43, Update 2200, Cost = 13,0093
Epoch 45, Update 2300, Cost = 14,1983
Epoch 47, Update 2400, Cost = 12,9891
Epoch 49, Update 2500, Cost = 18,9618
Epoch 50, Update 2600, Cost = 12,2912
Epoch 52, Update 2700, Cost = 12,3812
Epoch 54, Update 2800, Cost = 12,3731
Epoch 56, Update 2900, Cost = 12,1950
Epoch 58, Update 3000, Cost = 12,4011
Epoch 60, Update 3100, Cost = 12,4430
Epoch 62, Update 3200, Cost = 12,4558
Epoch 64, Update 3300, Cost = 12,3132
Epoch 66, Update 3400, Cost = 11,7202
Epoch 68, Update 3500, Cost = 11,2726
Epoch 70, Update 3600, Cost = 11,4279
Epoch 72, Update 3700, Cost = 11,3189
Epoch 74, Update 3800, Cost = 11,3157
Epoch 76, Update 3900, Cost = 11,0648
Epoch 78, Update 4000, Cost = 10,8009
Epoch 80, Update 4100, Cost = 10,8878
Epoch 82, Update 4200, Cost = 11,2118
Epoch 84, Update 4300, Cost = 11,7147
Epoch 86, Update 4400, Cost = 11,8365
Epoch 88, Update 4500, Cost = 12,1710
Epoch 90, Update 4600, Cost = 12,1169
Epoch 92, Update 4700, Cost = 12,3180
Epoch 94, Update 4800, Cost = 13,2608
Epoch 96, Update 4900, Cost = 13,9250
Epoch 98, Update 5000, Cost = 14,2095
Epoch 99, Update 5100, Cost = 12,2265
Epoch 101, Update 5200, Cost = 12,2970
Epoch 103, Update 5300, Cost = 12,3701
Epoch 105, Update 5400, Cost = 12,1472
Epoch 107, Update 5500, Cost = 12,3090
Epoch 109, Update 5600, Cost = 12,4080
Epoch 111, Update 5700, Cost = 12,4849
Epoch 113, Update 5800, Cost = 12,4343
Epoch 115, Update 5900, Cost = 12,2518
Epoch 117, Update 6000, Cost = 11,2659
Epoch 119, Update 6100, Cost = 11,3077
Epoch 121, Update 6200, Cost = 11,3957
Epoch 123, Update 6300, Cost = 11,3703
Epoch 125, Update 6400, Cost = 11,1992
Epoch 127, Update 6500, Cost = 10,8920
Epoch 129, Update 6600, Cost = 10,8220
Epoch 131, Update 6700, Cost = 10,9526
Epoch 133, Update 6800, Cost = 11,4211
Epoch 135, Update 6900, Cost = 11,7530
Epoch 137, Update 7000, Cost = 12,1670
Epoch 139, Update 7100, Cost = 12,2272
Epoch 141, Update 7200, Cost = 11,9832
Epoch 143, Update 7300, Cost = 13,0085
Epoch 145, Update 7400, Cost = 14,1978
Epoch 147, Update 7500, Cost = 12,9888
Epoch 149, Update 7600, Cost = 18,9616
Epoch 150, Update 7700, Cost = 12,2911
Epoch 152, Update 7800, Cost = 12,3811
Epoch 154, Update 7900, Cost = 12,3730
Epoch 156, Update 8000, Cost = 12,1950
Epoch 158, Update 8100, Cost = 12,4010
Epoch 160, Update 8200, Cost = 12,4430
Epoch 162, Update 8300, Cost = 12,4558
Epoch 164, Update 8400, Cost = 12,3132
Epoch 166, Update 8500, Cost = 11,7202
Epoch 168, Update 8600, Cost = 11,2726
Epoch 170, Update 8700, Cost = 11,4279
Epoch 172, Update 8800, Cost = 11,3189
Epoch 174, Update 8900, Cost = 11,3157
Epoch 176, Update 9000, Cost = 11,0648
Epoch 178, Update 9100, Cost = 10,8009
Epoch 180, Update 9200, Cost = 10,8878
Epoch 182, Update 9300, Cost = 11,2118
Epoch 184, Update 9400, Cost = 11,7147
Epoch 186, Update 9500, Cost = 11,8365
Epoch 188, Update 9600, Cost = 12,1710
Epoch 190, Update 9700, Cost = 12,1169
Epoch 192, Update 9800, Cost = 12,3180
Epoch 194, Update 9900, Cost = 17,5473
Epoch 196, Update 10000, Cost = 18,7538
Epoch 198, Update 10100, Cost = 18,4640
Epoch 199, Update 10200, Cost = 13,6652

Translations:
<s> È è ? </s>
<s> È è è </s>
<s> Cosa è </s>

             */

            Console.ReadLine();
        }
    }
}
