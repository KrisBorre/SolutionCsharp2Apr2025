using Seq2SeqSharp;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Enums;
using Seq2SeqSharp.LearningRate;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;

// ChatGPT, modified
namespace ConsoleDutchItalianSeq2Seq09Jul2025
{
    internal class Program
    {
        static void Main(string[] args)
        {
            string srcLang = "NL";
            string tgtLang = "IT";
            string modelFilePath = "nl2it.model";

            var trainData = new List<(string src, string tgt)>
            {
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
                MaxEpochNum = 70, // 60, // 200,
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
                "Veel geluk",
                "Ik wil reserveren",
                "Ik wil graag betalen",
                "Ik mis je",
                "Wat raad je aan ?",
                "Ik wil betalen",
                "Ik koop het hotel"
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

            /*
 Epoch 6, Update 100, Cost = 4,2327
Epoch 13, Update 200, Cost = 1,9048
Epoch 19, Update 300, Cost = 0,3170
Epoch 26, Update 400, Cost = 0,1090
Epoch 33, Update 500, Cost = 0,0457
Epoch 39, Update 600, Cost = 0,0243
Epoch 46, Update 700, Cost = 0,0170
Epoch 53, Update 800, Cost = 0,0117
Epoch 59, Update 900, Cost = 0,0079
Epoch 66, Update 1000, Cost = 3,9229
Epoch 73, Update 1100, Cost = 2,4367
Epoch 79, Update 1200, Cost = 0,8428
Epoch 86, Update 1300, Cost = 0,9199
Epoch 93, Update 1400, Cost = 1,0356
Epoch 99, Update 1500, Cost = 0,4728
Epoch 106, Update 1600, Cost = 0,6221
Epoch 113, Update 1700, Cost = 0,7904
Epoch 119, Update 1800, Cost = 0,3944
Epoch 126, Update 1900, Cost = 0,5052
Epoch 133, Update 2000, Cost = 0,6677
Epoch 139, Update 2100, Cost = 0,3431
Epoch 146, Update 2200, Cost = 0,4849
Epoch 153, Update 2300, Cost = 0,6491
Epoch 159, Update 2400, Cost = 0,3364
Epoch 166, Update 2500, Cost = 0,4786
Epoch 173, Update 2600, Cost = 0,6434
Epoch 179, Update 2700, Cost = 0,3344
Epoch 186, Update 2800, Cost = 0,4761
Epoch 193, Update 2900, Cost = 0,6409
Epoch 199, Update 3000, Cost = 0,3334

Translations:
<s> Buona fortuna </s>
<s> Vorrei prenotare </s>
<s> Vorrei pagare </s>
<s> Mi manchi </s>
<s> Cosa consigli ? </s>
<s> Vorrei pagare </s>
<s> Sto cercando l'hotel </s>
             */

            /*
             Epoch 6, Update 100, Cost = 4,8374
Epoch 13, Update 200, Cost = 1,2020
Epoch 19, Update 300, Cost = 0,3602
Epoch 26, Update 400, Cost = 0,1305
Epoch 33, Update 500, Cost = 0,0593
Epoch 39, Update 600, Cost = 0,0377
Epoch 46, Update 700, Cost = 0,0265
Epoch 53, Update 800, Cost = 0,0181
Epoch 59, Update 900, Cost = 0,0130
Epoch 66, Update 1000, Cost = 4,4542
Epoch 73, Update 1100, Cost = 3,8438
Epoch 79, Update 1200, Cost = 1,2238
Epoch 86, Update 1300, Cost = 1,3699
Epoch 93, Update 1400, Cost = 1,7232
Epoch 99, Update 1500, Cost = 0,7150
Epoch 106, Update 1600, Cost = 0,9474
Epoch 113, Update 1700, Cost = 1,3436
Epoch 119, Update 1800, Cost = 0,6029
Epoch 126, Update 1900, Cost = 0,7613
Epoch 133, Update 2000, Cost = 1,1229
Epoch 139, Update 2100, Cost = 0,5181
Epoch 146, Update 2200, Cost = 0,7312
Epoch 153, Update 2300, Cost = 1,0946
Epoch 159, Update 2400, Cost = 0,5084
Epoch 166, Update 2500, Cost = 0,7220
Epoch 173, Update 2600, Cost = 1,0858
Epoch 179, Update 2700, Cost = 0,5054
Epoch 186, Update 2800, Cost = 0,7192
Epoch 193, Update 2900, Cost = 1,0815
Epoch 199, Update 3000, Cost = 0,5038

Translations:
<s> Buona fortuna </s>
<s> Vorrei prenotare </s>
<s> Vorrei pagare </s>
<s> Mi manchi </s>
<s> Cosa consigli ? </s>
<s> Vorrei prenotare </s>
<s> Compro del pane </s>
             */

            /*
             Epoch 6, Update 100, Cost = 3,5995
Epoch 13, Update 200, Cost = 0,9531
Epoch 19, Update 300, Cost = 0,3217
Epoch 26, Update 400, Cost = 0,1704
Epoch 33, Update 500, Cost = 0,0760
Epoch 39, Update 600, Cost = 0,0363
Epoch 46, Update 700, Cost = 0,0256
Epoch 53, Update 800, Cost = 0,0186
Epoch 59, Update 900, Cost = 0,0111
Epoch 66, Update 1000, Cost = 4,8098
Epoch 73, Update 1100, Cost = 4,3453
Epoch 79, Update 1200, Cost = 1,1758
Epoch 86, Update 1300, Cost = 1,2718
Epoch 93, Update 1400, Cost = 1,9044
Epoch 99, Update 1500, Cost = 0,6723
Epoch 106, Update 1600, Cost = 0,8743
Epoch 113, Update 1700, Cost = 1,4629
Epoch 119, Update 1800, Cost = 0,5616
Epoch 126, Update 1900, Cost = 0,6862
Epoch 133, Update 2000, Cost = 1,1566
Epoch 139, Update 2100, Cost = 0,4691
Epoch 146, Update 2200, Cost = 0,6532
Epoch 153, Update 2300, Cost = 1,1258
Epoch 159, Update 2400, Cost = 0,4574
Epoch 166, Update 2500, Cost = 0,6437
Epoch 173, Update 2600, Cost = 1,1161
Epoch 179, Update 2700, Cost = 0,4539
Epoch 186, Update 2800, Cost = 0,6408
Epoch 193, Update 2900, Cost = 1,1109
Epoch 199, Update 3000, Cost = 0,4521

Translations:
<s> Buona fortuna </s>
<s> Vorrei prenotare </s>
<s> Vorrei pagare </s>
<s> Mi manchi </s>
<s> Cosa consigli ? </s>
<s> Vorrei pagare </s>
<s> Sto cercando l'hotel </s>
             */


            /*
             Epoch 6, Update 100, Cost = 4,0591
Epoch 13, Update 200, Cost = 1,9981
Epoch 19, Update 300, Cost = 0,2174
Epoch 26, Update 400, Cost = 0,3076
Epoch 33, Update 500, Cost = 0,0421
Epoch 39, Update 600, Cost = 0,0285
Epoch 46, Update 700, Cost = 0,0197
Epoch 53, Update 800, Cost = 0,0120
Epoch 59, Update 900, Cost = 0,0094

Translations:
<s> Buona fortuna </s>
<s> Vorrei </s>
<s> Vorrei pagare </s>
<s> Mi manchi </s>
<s> Ci vediamo </s>
<s> Vorrei pagare </s>
<s> Sto cercando </s>
             */

            /*
             Epoch 6, Update 100, Cost = 3,3961
Epoch 13, Update 200, Cost = 1,1321
Epoch 19, Update 300, Cost = 0,5279
Epoch 26, Update 400, Cost = 0,4890
Epoch 33, Update 500, Cost = 0,0788
Epoch 39, Update 600, Cost = 0,0424
Epoch 46, Update 700, Cost = 0,0294
Epoch 53, Update 800, Cost = 0,0212
Epoch 59, Update 900, Cost = 0,0135

Translations:
<s> Buona fortuna </s>
<s> Mi </s>
<s> Mi </s>
<s> Mi manchi </s>
<s> Cosa consigli ? </s>
<s> Mi </s>
<s> Sto cercando l'hotel </s>
             */

            /*
             Epoch 6, Update 100, Cost = 5,1076
Epoch 13, Update 200, Cost = 1,5285
Epoch 19, Update 300, Cost = 0,4531
Epoch 26, Update 400, Cost = 0,1015
Epoch 33, Update 500, Cost = 0,0420
Epoch 39, Update 600, Cost = 0,0225
Epoch 46, Update 700, Cost = 0,0155
Epoch 53, Update 800, Cost = 0,0109
Epoch 59, Update 900, Cost = 0,0072

Translations:
<s> Buona fortuna </s>
<s> Mi </s>
<s> Mi </s>
<s> Mi manchi </s>
<s> Cosa consigli </s>
<s> Mi manchi </s>
<s> Mi cercando </s>
             */


            /*
             Epoch 6, Update 100, Cost = 4,2905
Epoch 13, Update 200, Cost = 2,5463
Epoch 19, Update 300, Cost = 0,3025
Epoch 26, Update 400, Cost = 0,1035
Epoch 33, Update 500, Cost = 0,0515
Epoch 39, Update 600, Cost = 0,0333
Epoch 46, Update 700, Cost = 0,0250
Epoch 53, Update 800, Cost = 0,0165
Epoch 59, Update 900, Cost = 3,8544
Epoch 66, Update 1000, Cost = 2,2556

Translations:
<s> Buona fortuna </s>
<s> Vorrei prenotare </s>
<s> Vorrei pagare </s>
<s> Mi manchi </s>
<s> Cosa consigli ? </s>
<s> Vorrei pagare </s>
<s> Sto cercando l'hotel </s>
             */

            /*
             Epoch 6, Update 100, Cost = 2,5303
Epoch 13, Update 200, Cost = 0,7459
Epoch 19, Update 300, Cost = 0,2195
Epoch 26, Update 400, Cost = 0,0708
Epoch 33, Update 500, Cost = 0,0328
Epoch 39, Update 600, Cost = 0,0175
Epoch 46, Update 700, Cost = 0,0121
Epoch 53, Update 800, Cost = 0,0085
Epoch 59, Update 900, Cost = 1,4844
Epoch 66, Update 1000, Cost = 0,8684

Translations:
<s> Buona fortuna </s>
<s> Vorrei prenotare </s>
<s> Vorrei pagare </s>
<s> Mi manchi </s>
<s> Cosa consigli ? </s>
<s> Vorrei prenotare </s>
<s> Compro del l'hotel </s>

             */

            /*
             Epoch 6, Update 100, Cost = 3,5738
Epoch 13, Update 200, Cost = 0,9005
Epoch 19, Update 300, Cost = 0,5599
Epoch 26, Update 400, Cost = 0,1318
Epoch 33, Update 500, Cost = 0,0681
Epoch 39, Update 600, Cost = 0,0386
Epoch 46, Update 700, Cost = 0,0283
Epoch 53, Update 800, Cost = 0,0203
Epoch 59, Update 900, Cost = 0,0132
Epoch 66, Update 1000, Cost = 2,1688

Translations:
<s> Buona fortuna </s>
<s> Vorrei prenotare </s>
<s> Vorrei pagare </s>
<s> Mi manchi </s>
<s> Cosa consigli ? </s>
<s> Vorrei pagare </s>
<s> Sto cercando l'hotel </s>

             */

            Console.ReadLine();
        }
    }
}
