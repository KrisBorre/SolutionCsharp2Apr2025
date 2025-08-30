using Seq2SeqSharp;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Enums;
using Seq2SeqSharp.LearningRate;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;

// ChatGPT, aangepast
namespace ConsoleDutchItalianSeq2Seq4Jun2025
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
                HiddenSize = 64,
                SrcEmbeddingDim = 64,
                TgtEmbeddingDim = 64,
                MaxEpochNum = 200,
                MaxTokenSizePerBatch = 10,
                ValMaxTokenSizePerBatch = 10,
                MaxSrcSentLength = 20,
                MaxTgtSentLength = 20,
                PaddingType = PaddingEnums.AllowPadding,
                TooLongSequence = TooLongSequence.Ignore,
                ProcessorType = ProcessorTypeEnums.CPU,
                ModelFilePath = "nl2it_epoch200.model",
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
            opts.ModelFilePath = "nl2it_epoch200.model.trained";
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
            Epoch 1, Update 100, Cost = 18,5407
Epoch 3, Update 200, Cost = 17,4562
Epoch 5, Update 300, Cost = 16,6895
Epoch 7, Update 400, Cost = 16,6149
Epoch 9, Update 500, Cost = 16,5366
Epoch 11, Update 600, Cost = 16,4810
Epoch 13, Update 700, Cost = 16,3317
Epoch 15, Update 800, Cost = 16,1124
Epoch 17, Update 900, Cost = 14,8165
Epoch 19, Update 1000, Cost = 14,8683
Epoch 21, Update 1100, Cost = 15,0173
Epoch 23, Update 1200, Cost = 14,9568
Epoch 25, Update 1300, Cost = 14,7660
Epoch 27, Update 1400, Cost = 14,5045
Epoch 29, Update 1500, Cost = 14,4383
Epoch 31, Update 1600, Cost = 14,6333
Epoch 33, Update 1700, Cost = 15,3748
Epoch 35, Update 1800, Cost = 15,8758
Epoch 37, Update 1900, Cost = 16,4769
Epoch 39, Update 2000, Cost = 16,7276
Epoch 41, Update 2100, Cost = 16,5738
Epoch 43, Update 2200, Cost = 17,8934
Epoch 45, Update 2300, Cost = 19,2411
Epoch 47, Update 2400, Cost = 17,2055
Epoch 49, Update 2500, Cost = 22,1371
Epoch 50, Update 2600, Cost = 15,8589
Epoch 52, Update 2700, Cost = 16,0192
Epoch 54, Update 2800, Cost = 16,0856
Epoch 56, Update 2900, Cost = 15,9350
Epoch 58, Update 3000, Cost = 16,1683
Epoch 60, Update 3100, Cost = 16,2046
Epoch 62, Update 3200, Cost = 16,2008
Epoch 64, Update 3300, Cost = 16,0800
Epoch 66, Update 3400, Cost = 15,3708
Epoch 68, Update 3500, Cost = 14,7596
Epoch 70, Update 3600, Cost = 15,0131
Epoch 72, Update 3700, Cost = 14,8744
Epoch 74, Update 3800, Cost = 14,8555
Epoch 76, Update 3900, Cost = 14,6366
Epoch 78, Update 4000, Cost = 14,4267
Epoch 80, Update 4100, Cost = 14,5343
Epoch 82, Update 4200, Cost = 15,0263
Epoch 84, Update 4300, Cost = 15,8456
Epoch 86, Update 4400, Cost = 16,0124
Epoch 88, Update 4500, Cost = 16,6531
Epoch 90, Update 4600, Cost = 16,7394
Epoch 92, Update 4700, Cost = 16,9229
Epoch 94, Update 4800, Cost = 18,2661
Epoch 96, Update 4900, Cost = 18,5863
Epoch 98, Update 5000, Cost = 18,1281
Epoch 99, Update 5100, Cost = 15,7437
Epoch 101, Update 5200, Cost = 15,9027
Epoch 103, Update 5300, Cost = 16,0267
Epoch 105, Update 5400, Cost = 15,8333
Epoch 107, Update 5500, Cost = 16,0721
Epoch 109, Update 5600, Cost = 16,1885
Epoch 111, Update 5700, Cost = 16,2549
Epoch 113, Update 5800, Cost = 16,1848
Epoch 115, Update 5900, Cost = 16,0161
Epoch 117, Update 6000, Cost = 14,7539
Epoch 119, Update 6100, Cost = 14,8267
Epoch 121, Update 6200, Cost = 14,9890
Epoch 123, Update 6300, Cost = 14,9375
Epoch 125, Update 6400, Cost = 14,7530
Epoch 127, Update 6500, Cost = 14,4958
Epoch 129, Update 6600, Cost = 14,4326
Epoch 131, Update 6700, Cost = 14,6294
Epoch 133, Update 6800, Cost = 15,3721
Epoch 135, Update 6900, Cost = 15,8739
Epoch 137, Update 7000, Cost = 16,4756
Epoch 139, Update 7100, Cost = 16,7267
Epoch 141, Update 7200, Cost = 16,5732
Epoch 143, Update 7300, Cost = 17,8931
Epoch 145, Update 7400, Cost = 19,2408
Epoch 147, Update 7500, Cost = 17,2053
Epoch 149, Update 7600, Cost = 22,1371
Epoch 150, Update 7700, Cost = 15,8589
Epoch 152, Update 7800, Cost = 16,0191
Epoch 154, Update 7900, Cost = 16,0856
Epoch 156, Update 8000, Cost = 15,9350
Epoch 158, Update 8100, Cost = 16,1683
Epoch 160, Update 8200, Cost = 16,2046
Epoch 162, Update 8300, Cost = 16,2008
Epoch 164, Update 8400, Cost = 16,0800
Epoch 166, Update 8500, Cost = 15,3708
Epoch 168, Update 8600, Cost = 14,7596
Epoch 170, Update 8700, Cost = 15,0131
Epoch 172, Update 8800, Cost = 14,8744
Epoch 174, Update 8900, Cost = 14,8555
Epoch 176, Update 9000, Cost = 14,6366
Epoch 178, Update 9100, Cost = 14,4267
Epoch 180, Update 9200, Cost = 14,5343
Epoch 182, Update 9300, Cost = 15,0263
Epoch 184, Update 9400, Cost = 15,8456
Epoch 186, Update 9500, Cost = 16,0124
Epoch 188, Update 9600, Cost = 16,6531
Epoch 190, Update 9700, Cost = 16,7394
Epoch 192, Update 9800, Cost = 16,9229
Epoch 194, Update 9900, Cost = 18,2661
Epoch 196, Update 10000, Cost = 18,5863
Epoch 198, Update 10100, Cost = 18,1281
Epoch 199, Update 10200, Cost = 15,7437

Translations:
<s> È </s>
<s> </s>
<s> È </s>



                        */

            /*
             Epoch 1, Update 100, Cost = 18,7519
Epoch 3, Update 200, Cost = 17,5708
Epoch 5, Update 300, Cost = 16,7757
Epoch 7, Update 400, Cost = 16,6900
Epoch 9, Update 500, Cost = 16,6393
Epoch 11, Update 600, Cost = 16,5884
Epoch 13, Update 700, Cost = 16,4215
Epoch 15, Update 800, Cost = 16,1775
Epoch 17, Update 900, Cost = 14,9020
Epoch 19, Update 1000, Cost = 14,9548
Epoch 21, Update 1100, Cost = 15,1036
Epoch 23, Update 1200, Cost = 15,0604
Epoch 25, Update 1300, Cost = 14,8518
Epoch 27, Update 1400, Cost = 14,5376
Epoch 29, Update 1500, Cost = 14,4613
Epoch 31, Update 1600, Cost = 14,6037
Epoch 33, Update 1700, Cost = 15,3623
Epoch 35, Update 1800, Cost = 15,8171
Epoch 37, Update 1900, Cost = 16,5059
Epoch 39, Update 2000, Cost = 16,6712
Epoch 41, Update 2100, Cost = 16,4852
Epoch 43, Update 2200, Cost = 17,8968
Epoch 45, Update 2300, Cost = 19,1257
Epoch 47, Update 2400, Cost = 16,6745
Epoch 49, Update 2500, Cost = 21,4722
Epoch 50, Update 2600, Cost = 15,9196
Epoch 52, Update 2700, Cost = 16,0698
Epoch 54, Update 2800, Cost = 16,1231
Epoch 56, Update 2900, Cost = 15,9935
Epoch 58, Update 3000, Cost = 16,2445
Epoch 60, Update 3100, Cost = 16,3276
Epoch 62, Update 3200, Cost = 16,3050
Epoch 64, Update 3300, Cost = 16,1523
Epoch 66, Update 3400, Cost = 15,4693
Epoch 68, Update 3500, Cost = 14,8539
Epoch 70, Update 3600, Cost = 15,1024
Epoch 72, Update 3700, Cost = 14,9695
Epoch 74, Update 3800, Cost = 14,9552
Epoch 76, Update 3900, Cost = 14,6861
Epoch 78, Update 4000, Cost = 14,4640
Epoch 80, Update 4100, Cost = 14,5251
Epoch 82, Update 4200, Cost = 15,0055
Epoch 84, Update 4300, Cost = 15,8558
Epoch 86, Update 4400, Cost = 15,9806
Epoch 88, Update 4500, Cost = 16,5694
Epoch 90, Update 4600, Cost = 16,7214
Epoch 92, Update 4700, Cost = 16,8697
Epoch 94, Update 4800, Cost = 18,1665
Epoch 96, Update 4900, Cost = 18,1998
Epoch 98, Update 5000, Cost = 17,4777
Epoch 99, Update 5100, Cost = 15,7946
Epoch 101, Update 5200, Cost = 15,9617
Epoch 103, Update 5300, Cost = 16,0662
Epoch 105, Update 5400, Cost = 15,8830
Epoch 107, Update 5500, Cost = 16,1302
Epoch 109, Update 5600, Cost = 16,2853
Epoch 111, Update 5700, Cost = 16,3591
Epoch 113, Update 5800, Cost = 16,2738
Epoch 115, Update 5900, Cost = 16,0799
Epoch 117, Update 6000, Cost = 14,8394
Epoch 119, Update 6100, Cost = 14,9129
Epoch 121, Update 6200, Cost = 15,0748
Epoch 123, Update 6300, Cost = 15,0407
Epoch 125, Update 6400, Cost = 14,8387
Epoch 127, Update 6500, Cost = 14,5288
Epoch 129, Update 6600, Cost = 14,4554
Epoch 131, Update 6700, Cost = 14,5997
Epoch 133, Update 6800, Cost = 15,3595
Epoch 135, Update 6900, Cost = 15,8151
Epoch 137, Update 7000, Cost = 16,5046
Epoch 139, Update 7100, Cost = 16,6703
Epoch 141, Update 7200, Cost = 16,4846
Epoch 143, Update 7300, Cost = 17,8964
Epoch 145, Update 7400, Cost = 19,1254
Epoch 147, Update 7500, Cost = 16,6743
Epoch 149, Update 7600, Cost = 21,4721
Epoch 150, Update 7700, Cost = 15,9196
Epoch 152, Update 7800, Cost = 16,0698
Epoch 154, Update 7900, Cost = 16,1231
Epoch 156, Update 8000, Cost = 15,9935
Epoch 158, Update 8100, Cost = 16,2445
Epoch 160, Update 8200, Cost = 16,3276
Epoch 162, Update 8300, Cost = 16,3050
Epoch 164, Update 8400, Cost = 16,1523
Epoch 166, Update 8500, Cost = 15,4693
Epoch 168, Update 8600, Cost = 14,8539
Epoch 170, Update 8700, Cost = 15,1024
Epoch 172, Update 8800, Cost = 14,9695
Epoch 174, Update 8900, Cost = 14,9552
Epoch 176, Update 9000, Cost = 14,6861
Epoch 178, Update 9100, Cost = 14,4640
Epoch 180, Update 9200, Cost = 14,5251
Epoch 182, Update 9300, Cost = 15,0055
Epoch 184, Update 9400, Cost = 15,8558
Epoch 186, Update 9500, Cost = 15,9806
Epoch 188, Update 9600, Cost = 16,5694
Epoch 190, Update 9700, Cost = 16,7214
Epoch 192, Update 9800, Cost = 16,8697
Epoch 194, Update 9900, Cost = 18,1665
Epoch 196, Update 10000, Cost = 18,1998
Epoch 198, Update 10100, Cost = 17,4777
Epoch 199, Update 10200, Cost = 15,7946

Translations:
<s> Sono </s>
<s> </s>
<s> Sono </s>
            */

            Console.ReadLine();
        }
    }
}
