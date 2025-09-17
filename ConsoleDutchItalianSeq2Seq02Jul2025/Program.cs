using Seq2SeqSharp;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Enums;
using Seq2SeqSharp.LearningRate;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;

// ChatGPT, modified
namespace ConsoleDutchItalianSeq2Seq02Jul2025
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
                ("Ik ben moe", "Sono stanco")
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
                HiddenSize = 64,
                SrcEmbeddingDim = 64,
                TgtEmbeddingDim = 64,
                MaxEpochNum = 500,
                MaxTokenSizePerBatch = 10,
                ValMaxTokenSizePerBatch = 10,
                MaxSrcSentLength = 20,
                MaxTgtSentLength = 20,
                PaddingType = PaddingEnums.AllowPadding,
                TooLongSequence = TooLongSequence.Ignore,
                ProcessorType = ProcessorTypeEnums.CPU,
                ModelFilePath = "nl2it_epoch500.model",
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
            opts.ModelFilePath = "nl2it_epoch500.model.trained";
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
                decodingOptions: opts.CreateDecodingOptions(),
                srcSpmPath: null,
                tgtSpmPath: null); // We are not using SentencePiece

            Console.WriteLine("\nTranslations:");
            foreach (var line in File.ReadLines(testOutputPath))
            {
                Console.WriteLine(line);
            }

            /*
            Epoch 16, Update 100, Cost = 5,3607
Epoch 33, Update 200, Cost = 2,8728
Epoch 49, Update 300, Cost = 2,1056
Epoch 66, Update 400, Cost = 1,6184
Epoch 83, Update 500, Cost = 1,4097
Epoch 99, Update 600, Cost = 1,2952
Epoch 116, Update 700, Cost = 1,1810
Epoch 133, Update 800, Cost = 1,1419
Epoch 149, Update 900, Cost = 1,1120
Epoch 166, Update 1000, Cost = 1,0690
Epoch 183, Update 1100, Cost = 1,0679
Epoch 199, Update 1200, Cost = 1,0593
Epoch 216, Update 1300, Cost = 1,0361
Epoch 233, Update 1400, Cost = 1,0461
Epoch 249, Update 1500, Cost = 1,0438
Epoch 266, Update 1600, Cost = 1,0263
Epoch 283, Update 1700, Cost = 1,0394
Epoch 299, Update 1800, Cost = 1,0390
Epoch 316, Update 1900, Cost = 1,0233
Epoch 333, Update 2000, Cost = 1,0373
Epoch 349, Update 2100, Cost = 1,0375
Epoch 366, Update 2200, Cost = 1,0223
Epoch 383, Update 2300, Cost = 1,0367
Epoch 399, Update 2400, Cost = 1,0371
Epoch 416, Update 2500, Cost = 1,0220
Epoch 433, Update 2600, Cost = 1,0365
Epoch 449, Update 2700, Cost = 1,0369
Epoch 466, Update 2800, Cost = 1,0220
Epoch 483, Update 2900, Cost = 1,0365
Epoch 499, Update 3000, Cost = 1,0369

Translations:
<s> Che ore sono </s>
<s> Questa è la mia casa </s>
<s> Questa </s>
             */


            /*
             Epoch 16, Update 100, Cost = 6,0251
Epoch 33, Update 200, Cost = 3,4728
Epoch 49, Update 300, Cost = 2,3415
Epoch 66, Update 400, Cost = 1,8798
Epoch 83, Update 500, Cost = 1,8505
Epoch 99, Update 600, Cost = 1,4580
Epoch 116, Update 700, Cost = 1,3750
Epoch 133, Update 800, Cost = 1,5192
Epoch 149, Update 900, Cost = 1,2537
Epoch 166, Update 1000, Cost = 1,2456
Epoch 183, Update 1100, Cost = 1,4270
Epoch 199, Update 1200, Cost = 1,1971
Epoch 216, Update 1300, Cost = 1,2089
Epoch 233, Update 1400, Cost = 1,4002
Epoch 249, Update 1500, Cost = 1,1804
Epoch 266, Update 1600, Cost = 1,1979
Epoch 283, Update 1700, Cost = 1,3920
Epoch 299, Update 1800, Cost = 1,1753
Epoch 316, Update 1900, Cost = 1,1945
Epoch 333, Update 2000, Cost = 1,3895
Epoch 349, Update 2100, Cost = 1,1737
Epoch 366, Update 2200, Cost = 1,1934
Epoch 383, Update 2300, Cost = 1,3887
Epoch 399, Update 2400, Cost = 1,1732
Epoch 416, Update 2500, Cost = 1,1931
Epoch 433, Update 2600, Cost = 1,3885
Epoch 449, Update 2700, Cost = 1,1730
Epoch 466, Update 2800, Cost = 1,1930
Epoch 483, Update 2900, Cost = 1,3884
Epoch 499, Update 3000, Cost = 1,1730

Translations:
<s> Che ore sono ? </s>
<s> Questa è la mia </s>
<s> Questa è la ? </s>
             */

            /*
             Epoch 16, Update 100, Cost = 5,6901
Epoch 33, Update 200, Cost = 3,2205
Epoch 49, Update 300, Cost = 2,1553
Epoch 66, Update 400, Cost = 1,6755
Epoch 83, Update 500, Cost = 1,5235
Epoch 99, Update 600, Cost = 1,3161
Epoch 116, Update 700, Cost = 1,2075
Epoch 133, Update 800, Cost = 1,2231
Epoch 149, Update 900, Cost = 1,1323
Epoch 166, Update 1000, Cost = 1,0928
Epoch 183, Update 1100, Cost = 1,1430
Epoch 199, Update 1200, Cost = 1,0813
Epoch 216, Update 1300, Cost = 1,0602
Epoch 233, Update 1400, Cost = 1,1199
Epoch 249, Update 1500, Cost = 1,0664
Epoch 266, Update 1600, Cost = 1,0505
Epoch 283, Update 1700, Cost = 1,1129
Epoch 299, Update 1800, Cost = 1,0618
Epoch 316, Update 1900, Cost = 1,0475
Epoch 333, Update 2000, Cost = 1,1108
Epoch 349, Update 2100, Cost = 1,0603
Epoch 366, Update 2200, Cost = 1,0466
Epoch 383, Update 2300, Cost = 1,1101
Epoch 399, Update 2400, Cost = 1,0599
Epoch 416, Update 2500, Cost = 1,0463
Epoch 433, Update 2600, Cost = 1,1099
Epoch 449, Update 2700, Cost = 1,0598
Epoch 466, Update 2800, Cost = 1,0462
Epoch 483, Update 2900, Cost = 1,1099
Epoch 499, Update 3000, Cost = 1,0598

Translations:
<s> Che ore sono ? </s>
<s> Questa è la mia un'insegnante un </s>
<s> Ti è Ti </s>
             */

            Console.ReadLine();
        }
    }
}
