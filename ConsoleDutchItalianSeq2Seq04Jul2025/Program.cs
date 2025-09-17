using Seq2SeqSharp;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Enums;
using Seq2SeqSharp.LearningRate;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;

// ChatGPT, modified
namespace ConsoleDutchItalianSeq2Seq04Jul2025
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
                MaxEpochNum = 800,
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
        Epoch 16, Update 100, Cost = 5,8773
Epoch 33, Update 200, Cost = 3,4808
Epoch 49, Update 300, Cost = 2,2909
Epoch 66, Update 400, Cost = 1,7430
Epoch 83, Update 500, Cost = 1,6576
Epoch 99, Update 600, Cost = 1,4106
Epoch 116, Update 700, Cost = 1,2620
Epoch 133, Update 800, Cost = 1,3286
Epoch 149, Update 900, Cost = 1,2120
Epoch 166, Update 1000, Cost = 1,1420
Epoch 183, Update 1100, Cost = 1,2414
Epoch 199, Update 1200, Cost = 1,1565
Epoch 216, Update 1300, Cost = 1,1076
Epoch 233, Update 1400, Cost = 1,2159
Epoch 249, Update 1500, Cost = 1,1400
Epoch 266, Update 1600, Cost = 1,0973
Epoch 283, Update 1700, Cost = 1,2082
Epoch 299, Update 1800, Cost = 1,1350
Epoch 316, Update 1900, Cost = 1,0942
Epoch 333, Update 2000, Cost = 1,2058
Epoch 349, Update 2100, Cost = 1,1334
Epoch 366, Update 2200, Cost = 1,0932
Epoch 383, Update 2300, Cost = 1,2050
Epoch 399, Update 2400, Cost = 1,1329
Epoch 416, Update 2500, Cost = 1,0929
Epoch 433, Update 2600, Cost = 1,2048
Epoch 449, Update 2700, Cost = 1,1328
Epoch 466, Update 2800, Cost = 1,0928
Epoch 483, Update 2900, Cost = 1,2048
Epoch 499, Update 3000, Cost = 1,1327
Epoch 516, Update 3100, Cost = 1,0928
Epoch 533, Update 3200, Cost = 1,2048
Epoch 549, Update 3300, Cost = 1,1327
Epoch 566, Update 3400, Cost = 1,0928
Epoch 583, Update 3500, Cost = 1,2048
Epoch 599, Update 3600, Cost = 1,1327
Epoch 616, Update 3700, Cost = 1,0928
Epoch 633, Update 3800, Cost = 1,2048
Epoch 649, Update 3900, Cost = 1,1327
Epoch 666, Update 4000, Cost = 1,0928
Epoch 683, Update 4100, Cost = 1,2048
Epoch 699, Update 4200, Cost = 1,1327
Epoch 716, Update 4300, Cost = 1,0928
Epoch 733, Update 4400, Cost = 1,2048
Epoch 749, Update 4500, Cost = 1,1327
Epoch 766, Update 4600, Cost = 1,0928
Epoch 783, Update 4700, Cost = 1,2048
Epoch 799, Update 4800, Cost = 1,1327

Translations:
<s> Che ore sono ? </s>
<s> Questa è la la </s>
<s> Questa </s>
             */

            /*
             Epoch 16, Update 100, Cost = 5,4370
Epoch 33, Update 200, Cost = 3,0176
Epoch 49, Update 300, Cost = 2,1583
Epoch 66, Update 400, Cost = 1,6461
Epoch 83, Update 500, Cost = 1,5360
Epoch 99, Update 600, Cost = 1,3097
Epoch 116, Update 700, Cost = 1,1958
Epoch 133, Update 800, Cost = 1,2518
Epoch 149, Update 900, Cost = 1,1256
Epoch 166, Update 1000, Cost = 1,0855
Epoch 183, Update 1100, Cost = 1,1764
Epoch 199, Update 1200, Cost = 1,0756
Epoch 216, Update 1300, Cost = 1,0546
Epoch 233, Update 1400, Cost = 1,1546
Epoch 249, Update 1500, Cost = 1,0610
Epoch 266, Update 1600, Cost = 1,0453
Epoch 283, Update 1700, Cost = 1,1479
Epoch 299, Update 1800, Cost = 1,0565
Epoch 316, Update 1900, Cost = 1,0425
Epoch 333, Update 2000, Cost = 1,1459
Epoch 349, Update 2100, Cost = 1,0550
Epoch 366, Update 2200, Cost = 1,0416
Epoch 383, Update 2300, Cost = 1,1452
Epoch 399, Update 2400, Cost = 1,0546
Epoch 416, Update 2500, Cost = 1,0413
Epoch 433, Update 2600, Cost = 1,1450
Epoch 449, Update 2700, Cost = 1,0545
Epoch 466, Update 2800, Cost = 1,0412
Epoch 483, Update 2900, Cost = 1,1450
Epoch 499, Update 3000, Cost = 1,0545
Epoch 516, Update 3100, Cost = 1,0412
Epoch 533, Update 3200, Cost = 1,1450
Epoch 549, Update 3300, Cost = 1,0545
Epoch 566, Update 3400, Cost = 1,0412
Epoch 583, Update 3500, Cost = 1,1450
Epoch 599, Update 3600, Cost = 1,0545
Epoch 616, Update 3700, Cost = 1,0412
Epoch 633, Update 3800, Cost = 1,1450
Epoch 649, Update 3900, Cost = 1,0545
Epoch 666, Update 4000, Cost = 1,0412
Epoch 683, Update 4100, Cost = 1,1450
Epoch 699, Update 4200, Cost = 1,0545
Epoch 716, Update 4300, Cost = 1,0412
Epoch 733, Update 4400, Cost = 1,1450
Epoch 749, Update 4500, Cost = 1,0545
Epoch 766, Update 4600, Cost = 1,0412
Epoch 783, Update 4700, Cost = 1,1450
Epoch 799, Update 4800, Cost = 1,0545

Translations:
<s> Che ore sono ? </s>
<s> Questa è la mia casa </s>
<s> Questo è è stai Che casa Che Che Questo Che Che Che Che Che Che Che Che Che Che Che Che Che Parlo Che </s>
             */


            /*
             Epoch 16, Update 100, Cost = 5,7131
Epoch 33, Update 200, Cost = 3,4329
Epoch 49, Update 300, Cost = 2,4463
Epoch 66, Update 400, Cost = 1,8241
Epoch 83, Update 500, Cost = 1,7665
Epoch 99, Update 600, Cost = 1,5176
Epoch 116, Update 700, Cost = 1,3503
Epoch 133, Update 800, Cost = 1,4489
Epoch 149, Update 900, Cost = 1,2998
Epoch 166, Update 1000, Cost = 1,2282
Epoch 183, Update 1100, Cost = 1,3624
Epoch 199, Update 1200, Cost = 1,2396
Epoch 216, Update 1300, Cost = 1,1935
Epoch 233, Update 1400, Cost = 1,3374
Epoch 249, Update 1500, Cost = 1,2219
Epoch 266, Update 1600, Cost = 1,1831
Epoch 283, Update 1700, Cost = 1,3299
Epoch 299, Update 1800, Cost = 1,2165
Epoch 316, Update 1900, Cost = 1,1799
Epoch 333, Update 2000, Cost = 1,3275
Epoch 349, Update 2100, Cost = 1,2148
Epoch 366, Update 2200, Cost = 1,1788
Epoch 383, Update 2300, Cost = 1,3268
Epoch 399, Update 2400, Cost = 1,2142
Epoch 416, Update 2500, Cost = 1,1785
Epoch 433, Update 2600, Cost = 1,3266
Epoch 449, Update 2700, Cost = 1,2141
Epoch 466, Update 2800, Cost = 1,1785
Epoch 483, Update 2900, Cost = 1,3265
Epoch 499, Update 3000, Cost = 1,2141
Epoch 516, Update 3100, Cost = 1,1785
Epoch 533, Update 3200, Cost = 1,3265
Epoch 549, Update 3300, Cost = 1,2141
Epoch 566, Update 3400, Cost = 1,1785
Epoch 583, Update 3500, Cost = 1,3265
Epoch 599, Update 3600, Cost = 1,2141
Epoch 616, Update 3700, Cost = 1,1785
Epoch 633, Update 3800, Cost = 1,3265
Epoch 649, Update 3900, Cost = 1,2141
Epoch 666, Update 4000, Cost = 1,1785
Epoch 683, Update 4100, Cost = 1,3265
Epoch 699, Update 4200, Cost = 1,2141
Epoch 716, Update 4300, Cost = 1,1785
Epoch 733, Update 4400, Cost = 1,3265
Epoch 749, Update 4500, Cost = 1,2141
Epoch 766, Update 4600, Cost = 1,1785
Epoch 783, Update 4700, Cost = 1,3265
Epoch 799, Update 4800, Cost = 1,2141

Translations:
<s> Che ore sono ? </s>
<s> Questa è la mia </s>
<s> Lei sono ? </s>
             */


            /*
             Epoch 16, Update 100, Cost = 6,0488
Epoch 33, Update 200, Cost = 3,2902
Epoch 49, Update 300, Cost = 2,3401
Epoch 66, Update 400, Cost = 1,8034
Epoch 83, Update 500, Cost = 1,5430
Epoch 99, Update 600, Cost = 1,4341
Epoch 116, Update 700, Cost = 1,3072
Epoch 133, Update 800, Cost = 1,2386
Epoch 149, Update 900, Cost = 1,2348
Epoch 166, Update 1000, Cost = 1,1827
Epoch 183, Update 1100, Cost = 1,1561
Epoch 199, Update 1200, Cost = 1,1802
Epoch 216, Update 1300, Cost = 1,1477
Epoch 233, Update 1400, Cost = 1,1322
Epoch 249, Update 1500, Cost = 1,1642
Epoch 266, Update 1600, Cost = 1,1371
Epoch 283, Update 1700, Cost = 1,1249
Epoch 299, Update 1800, Cost = 1,1592
Epoch 316, Update 1900, Cost = 1,1339
Epoch 333, Update 2000, Cost = 1,1226
Epoch 349, Update 2100, Cost = 1,1577
Epoch 366, Update 2200, Cost = 1,1329
Epoch 383, Update 2300, Cost = 1,1219
Epoch 399, Update 2400, Cost = 1,1572
Epoch 416, Update 2500, Cost = 1,1326
Epoch 433, Update 2600, Cost = 1,1217
Epoch 449, Update 2700, Cost = 1,1571
Epoch 466, Update 2800, Cost = 1,1325
Epoch 483, Update 2900, Cost = 1,1217
Epoch 499, Update 3000, Cost = 1,1571
Epoch 516, Update 3100, Cost = 1,1325
Epoch 533, Update 3200, Cost = 1,1217
Epoch 549, Update 3300, Cost = 1,1571
Epoch 566, Update 3400, Cost = 1,1325
Epoch 583, Update 3500, Cost = 1,1217
Epoch 599, Update 3600, Cost = 1,1571
Epoch 616, Update 3700, Cost = 1,1325
Epoch 633, Update 3800, Cost = 1,1217
Epoch 649, Update 3900, Cost = 1,1571
Epoch 666, Update 4000, Cost = 1,1325
Epoch 683, Update 4100, Cost = 1,1217
Epoch 699, Update 4200, Cost = 1,1571
Epoch 716, Update 4300, Cost = 1,1325
Epoch 733, Update 4400, Cost = 1,1217
Epoch 749, Update 4500, Cost = 1,1571
Epoch 766, Update 4600, Cost = 1,1325
Epoch 783, Update 4700, Cost = 1,1217
Epoch 799, Update 4800, Cost = 1,1571

Translations:
<s> Che ore sono ? </s>
<s> Questa è la mia </s>
<s> Ti </s>
             */

            Console.ReadLine();
        }
    }
}
