using Seq2SeqSharp;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Enums;
using Seq2SeqSharp.LearningRate;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;

// ChatGPT, aangepast
namespace ConsoleDutchItalianSeq2Seq6Jun2025
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
                HiddenSize = 256,
                SrcEmbeddingDim = 256,
                TgtEmbeddingDim = 256,
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
Epoch 1, Update 100, Cost = 16,6913
Epoch 3, Update 200, Cost = 12,9744
Epoch 5, Update 300, Cost = 10,1806
Epoch 7, Update 400, Cost = 8,6746
Epoch 9, Update 500, Cost = 7,9808
Epoch 11, Update 600, Cost = 7,2877
Epoch 13, Update 700, Cost = 6,8768
Epoch 15, Update 800, Cost = 6,5798
Epoch 17, Update 900, Cost = 5,6843
Epoch 19, Update 1000, Cost = 5,6952
Epoch 21, Update 1100, Cost = 5,6479
Epoch 23, Update 1200, Cost = 5,5898
Epoch 25, Update 1300, Cost = 5,4929
Epoch 27, Update 1400, Cost = 5,3554
Epoch 29, Update 1500, Cost = 10,4021
Epoch 31, Update 1600, Cost = 10,5258
Epoch 33, Update 1700, Cost = 11,0702
Epoch 35, Update 1800, Cost = 11,5857
Epoch 37, Update 1900, Cost = 12,4490
Epoch 39, Update 2000, Cost = 12,8023
Epoch 41, Update 2100, Cost = 13,0166
Epoch 43, Update 2200, Cost = 14,5515
Epoch 45, Update 2300, Cost = 15,8970
Epoch 47, Update 2400, Cost = 14,5248
Epoch 49, Update 2500, Cost = 20,7228
Epoch 50, Update 2600, Cost = 9,1713
Epoch 52, Update 2700, Cost = 9,3468
Epoch 54, Update 2800, Cost = 9,4543
Epoch 56, Update 2900, Cost = 9,3652
Epoch 58, Update 3000, Cost = 9,5895
Epoch 60, Update 3100, Cost = 9,7245
Epoch 62, Update 3200, Cost = 9,8598
Epoch 64, Update 3300, Cost = 9,8553
Epoch 66, Update 3400, Cost = 9,6080
Epoch 68, Update 3500, Cost = 9,3241
Epoch 70, Update 3600, Cost = 9,5865
Epoch 72, Update 3700, Cost = 9,6589
Epoch 74, Update 3800, Cost = 9,8266
Epoch 76, Update 3900, Cost = 9,8907
Epoch 78, Update 4000, Cost = 9,9146
Epoch 80, Update 4100, Cost = 10,0929
Epoch 82, Update 4200, Cost = 10,5652
Epoch 84, Update 4300, Cost = 11,1909
Epoch 86, Update 4400, Cost = 11,7177
Epoch 88, Update 4500, Cost = 12,4860
Epoch 90, Update 4600, Cost = 12,8504
Epoch 92, Update 4700, Cost = 13,4485
Epoch 94, Update 4800, Cost = 14,8162
Epoch 96, Update 4900, Cost = 15,3370
Epoch 98, Update 5000, Cost = 16,3227
Epoch 99, Update 5100, Cost = 9,0149
Epoch 101, Update 5200, Cost = 9,1585
Epoch 103, Update 5300, Cost = 9,3264
Epoch 105, Update 5400, Cost = 9,2646
Epoch 107, Update 5500, Cost = 9,4751
Epoch 109, Update 5600, Cost = 9,6505
Epoch 111, Update 5700, Cost = 9,8213
Epoch 113, Update 5800, Cost = 9,9261
Epoch 115, Update 5900, Cost = 9,9265
Epoch 117, Update 6000, Cost = 9,2632
Epoch 119, Update 6100, Cost = 9,4597
Epoch 121, Update 6200, Cost = 9,6631
Epoch 123, Update 6300, Cost = 9,7705
Epoch 125, Update 6400, Cost = 9,8868
Epoch 127, Update 6500, Cost = 9,8801
Epoch 129, Update 6600, Cost = 9,9919
Epoch 131, Update 6700, Cost = 10,2459
Epoch 133, Update 6800, Cost = 10,8442
Epoch 135, Update 6900, Cost = 11,3960
Epoch 137, Update 7000, Cost = 12,2800
Epoch 139, Update 7100, Cost = 12,6601
Epoch 141, Update 7200, Cost = 12,8965
Epoch 143, Update 7300, Cost = 14,4618
Epoch 145, Update 7400, Cost = 15,8573
Epoch 147, Update 7500, Cost = 14,4621
Epoch 149, Update 7600, Cost = 20,8274
Epoch 150, Update 7700, Cost = 9,0980
Epoch 152, Update 7800, Cost = 9,2717
Epoch 154, Update 7900, Cost = 9,3768
Epoch 156, Update 8000, Cost = 9,3647
Epoch 158, Update 8100, Cost = 9,5891
Epoch 160, Update 8200, Cost = 9,7242
Epoch 162, Update 8300, Cost = 9,8596
Epoch 164, Update 8400, Cost = 9,8551
Epoch 166, Update 8500, Cost = 9,6078
Epoch 168, Update 8600, Cost = 9,3239
Epoch 170, Update 8700, Cost = 9,5862
Epoch 172, Update 8800, Cost = 9,6587
Epoch 174, Update 8900, Cost = 9,8264
Epoch 176, Update 9000, Cost = 9,8905
Epoch 178, Update 9100, Cost = 9,9144
Epoch 180, Update 9200, Cost = 10,0927
Epoch 182, Update 9300, Cost = 10,5650
Epoch 184, Update 9400, Cost = 11,1909
Epoch 186, Update 9500, Cost = 11,7177
Epoch 188, Update 9600, Cost = 12,4860
Epoch 190, Update 9700, Cost = 12,8504
Epoch 192, Update 9800, Cost = 13,4485
Epoch 194, Update 9900, Cost = 14,8162
Epoch 196, Update 10000, Cost = 15,3370
Epoch 198, Update 10100, Cost = 16,3227
Epoch 199, Update 10200, Cost = 9,0149

Translations:
<s> Per costa ? ? </s>
<s> È è è </s>
<s> È è </s>

             */

            /*
             Vocabulary sizes: src = 196, tgt = 210
Epoch 1, Update 100, Cost = 17,3123
Epoch 3, Update 200, Cost = 14,8056
Epoch 5, Update 300, Cost = 12,0368
Epoch 7, Update 400, Cost = 10,6149
Epoch 9, Update 500, Cost = 9,9115
Epoch 11, Update 600, Cost = 9,3037
Epoch 13, Update 700, Cost = 8,7398
Epoch 15, Update 800, Cost = 8,4064
Epoch 17, Update 900, Cost = 7,4503
Epoch 19, Update 1000, Cost = 7,3947
Epoch 21, Update 1100, Cost = 7,3261
Epoch 23, Update 1200, Cost = 7,1748
Epoch 25, Update 1300, Cost = 7,0262
Epoch 27, Update 1400, Cost = 6,7543
Epoch 29, Update 1500, Cost = 6,5236
Epoch 31, Update 1600, Cost = 6,5559
Epoch 33, Update 1700, Cost = 6,7284
Epoch 35, Update 1800, Cost = 6,8786
Epoch 37, Update 1900, Cost = 7,0508
Epoch 39, Update 2000, Cost = 7,1491
Epoch 41, Update 2100, Cost = 6,9237
Epoch 43, Update 2200, Cost = 7,5009
Epoch 45, Update 2300, Cost = 8,0193
Epoch 47, Update 2400, Cost = 6,8404
Epoch 49, Update 2500, Cost = 10,3346
Epoch 50, Update 2600, Cost = 8,0186
Epoch 52, Update 2700, Cost = 8,0606
Epoch 54, Update 2800, Cost = 8,0217
Epoch 56, Update 2900, Cost = 7,8845
Epoch 58, Update 3000, Cost = 8,0017
Epoch 60, Update 3100, Cost = 7,9690
Epoch 62, Update 3200, Cost = 7,9588
Epoch 64, Update 3300, Cost = 7,9374
Epoch 66, Update 3400, Cost = 11,4526
Epoch 68, Update 3500, Cost = 11,2562
Epoch 70, Update 3600, Cost = 11,5073
Epoch 72, Update 3700, Cost = 11,5903
Epoch 74, Update 3800, Cost = 11,7753
Epoch 76, Update 3900, Cost = 11,8063
Epoch 78, Update 4000, Cost = 11,6844
Epoch 80, Update 4100, Cost = 11,9042
Epoch 82, Update 4200, Cost = 12,3251
Epoch 84, Update 4300, Cost = 13,0142
Epoch 86, Update 4400, Cost = 13,4944
Epoch 88, Update 4500, Cost = 14,2845
Epoch 90, Update 4600, Cost = 14,4644
Epoch 92, Update 4700, Cost = 15,0845
Epoch 94, Update 4800, Cost = 16,2200
Epoch 96, Update 4900, Cost = 17,0076
Epoch 98, Update 5000, Cost = 16,8634
Epoch 99, Update 5100, Cost = 10,8381
Epoch 101, Update 5200, Cost = 10,9879
Epoch 103, Update 5300, Cost = 11,1343
Epoch 105, Update 5400, Cost = 11,0623
Epoch 107, Update 5500, Cost = 11,2992
Epoch 109, Update 5600, Cost = 11,4703
Epoch 111, Update 5700, Cost = 11,6383
Epoch 113, Update 5800, Cost = 11,7302
Epoch 115, Update 5900, Cost = 11,8009
Epoch 117, Update 6000, Cost = 11,1251
Epoch 119, Update 6100, Cost = 11,3572
Epoch 121, Update 6200, Cost = 11,5840
Epoch 123, Update 6300, Cost = 11,7076
Epoch 125, Update 6400, Cost = 11,8360
Epoch 127, Update 6500, Cost = 11,7037
Epoch 129, Update 6600, Cost = 11,7630
Epoch 131, Update 6700, Cost = 12,0420
Epoch 133, Update 6800, Cost = 12,6505
Epoch 135, Update 6900, Cost = 13,2149
Epoch 137, Update 7000, Cost = 14,0381
Epoch 139, Update 7100, Cost = 14,4152
Epoch 141, Update 7200, Cost = 14,6000
Epoch 143, Update 7300, Cost = 15,9184
Epoch 145, Update 7400, Cost = 17,3323
Epoch 147, Update 7500, Cost = 15,8229
Epoch 149, Update 7600, Cost = 22,6175
Epoch 150, Update 7700, Cost = 10,9424
Epoch 152, Update 7800, Cost = 11,0997
Epoch 154, Update 7900, Cost = 11,1887
Epoch 156, Update 8000, Cost = 11,1750
Epoch 158, Update 8100, Cost = 11,4316
Epoch 160, Update 8200, Cost = 11,5276
Epoch 162, Update 8300, Cost = 11,6511
Epoch 164, Update 8400, Cost = 11,7834
Epoch 166, Update 8500, Cost = 11,4526
Epoch 168, Update 8600, Cost = 11,2562
Epoch 170, Update 8700, Cost = 11,5073
Epoch 172, Update 8800, Cost = 11,5903
Epoch 174, Update 8900, Cost = 11,7753
Epoch 176, Update 9000, Cost = 11,8063
Epoch 178, Update 9100, Cost = 11,6844
Epoch 180, Update 9200, Cost = 11,9041
Epoch 182, Update 9300, Cost = 12,3251
Epoch 184, Update 9400, Cost = 13,0142
Epoch 186, Update 9500, Cost = 13,4944
Epoch 188, Update 9600, Cost = 14,2845
Epoch 190, Update 9700, Cost = 14,4644
Epoch 192, Update 9800, Cost = 15,0845
Epoch 194, Update 9900, Cost = 16,2200
Epoch 196, Update 10000, Cost = 17,0076
Epoch 198, Update 10100, Cost = 16,8634
Epoch 199, Update 10200, Cost = 10,8381

Translations:
<s> Dove è ? </s>
<s> È è è casa </s>
<s> È è </s>

             */

            Console.ReadLine();
        }
    }
}
