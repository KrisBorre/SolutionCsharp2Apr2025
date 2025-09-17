using Seq2SeqSharp;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Enums;
using Seq2SeqSharp.LearningRate;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;

// ChatGPT, modified
namespace ConsoleDutchItalianSeq2Seq03Jul2025
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
                MaxEpochNum = 700,
                MaxTokenSizePerBatch = 10,
                ValMaxTokenSizePerBatch = 10,
                MaxSrcSentLength = 20,
                MaxTgtSentLength = 20,
                PaddingType = PaddingEnums.AllowPadding,
                TooLongSequence = TooLongSequence.Ignore,
                ProcessorType = ProcessorTypeEnums.CPU,
                ModelFilePath = "nl2it.model",
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
            opts.ModelFilePath = "nl2it.model.trained";
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
                //Console.WriteLine($"Deleted: {file}");
            }

            string[] files2 = Directory.GetFiles(rootPath, "nl2it.model.*");
            foreach (string file in files2)
            {
                File.Delete(file);
                //Console.WriteLine($"Deleted: {file}");
            }

            File.Delete(srcTrainFile);
            File.Delete(tgtTrainFile);
            File.Delete(testInputPath);
            File.Delete(testOutputPath);

            /*
            Epoch 16, Update 100, Cost = 5,4973
Epoch 33, Update 200, Cost = 2,8751
Epoch 49, Update 300, Cost = 2,1360
Epoch 66, Update 400, Cost = 1,7262
Epoch 83, Update 500, Cost = 1,5134
Epoch 99, Update 600, Cost = 1,3374
Epoch 116, Update 700, Cost = 1,2622
Epoch 133, Update 800, Cost = 1,2327
Epoch 149, Update 900, Cost = 1,1536
Epoch 166, Update 1000, Cost = 1,1422
Epoch 183, Update 1100, Cost = 1,1540
Epoch 199, Update 1200, Cost = 1,1013
Epoch 216, Update 1300, Cost = 1,1073
Epoch 233, Update 1400, Cost = 1,1307
Epoch 249, Update 1500, Cost = 1,0858
Epoch 266, Update 1600, Cost = 1,0969
Epoch 283, Update 1700, Cost = 1,1236
Epoch 299, Update 1800, Cost = 1,0810
Epoch 316, Update 1900, Cost = 1,0937
Epoch 333, Update 2000, Cost = 1,1214
Epoch 349, Update 2100, Cost = 1,0795
Epoch 366, Update 2200, Cost = 1,0926
Epoch 383, Update 2300, Cost = 1,1207
Epoch 399, Update 2400, Cost = 1,0791
Epoch 416, Update 2500, Cost = 1,0923
Epoch 433, Update 2600, Cost = 1,1205
Epoch 449, Update 2700, Cost = 1,0790
Epoch 466, Update 2800, Cost = 1,0923
Epoch 483, Update 2900, Cost = 1,1205
Epoch 499, Update 3000, Cost = 1,0789
Epoch 516, Update 3100, Cost = 1,0923
Epoch 533, Update 3200, Cost = 1,1205
Epoch 549, Update 3300, Cost = 1,0789
Epoch 566, Update 3400, Cost = 1,0923
Epoch 583, Update 3500, Cost = 1,1205
Epoch 599, Update 3600, Cost = 1,0789
Epoch 616, Update 3700, Cost = 1,0923
Epoch 633, Update 3800, Cost = 1,1205
Epoch 649, Update 3900, Cost = 1,0789
Epoch 666, Update 4000, Cost = 1,0923
Epoch 683, Update 4100, Cost = 1,1205
Epoch 699, Update 4200, Cost = 1,0789

Translations:
<s> Che ore sono ? </s>
<s> Questa è </s>
<s> amo amo </s>
             */

            /*
             Epoch 16, Update 100, Cost = 5,6541
Epoch 33, Update 200, Cost = 3,3529
Epoch 49, Update 300, Cost = 2,2958
Epoch 66, Update 400, Cost = 1,7493
Epoch 83, Update 500, Cost = 1,7366
Epoch 99, Update 600, Cost = 1,4245
Epoch 116, Update 700, Cost = 1,2863
Epoch 133, Update 800, Cost = 1,4182
Epoch 149, Update 900, Cost = 1,2274
Epoch 166, Update 1000, Cost = 1,1662
Epoch 183, Update 1100, Cost = 1,3282
Epoch 199, Update 1200, Cost = 1,1712
Epoch 216, Update 1300, Cost = 1,1314
Epoch 233, Update 1400, Cost = 1,3016
Epoch 249, Update 1500, Cost = 1,1546
Epoch 266, Update 1600, Cost = 1,1210
Epoch 283, Update 1700, Cost = 1,2935
Epoch 299, Update 1800, Cost = 1,1495
Epoch 316, Update 1900, Cost = 1,1178
Epoch 333, Update 2000, Cost = 1,2910
Epoch 349, Update 2100, Cost = 1,1479
Epoch 366, Update 2200, Cost = 1,1168
Epoch 383, Update 2300, Cost = 1,2902
Epoch 399, Update 2400, Cost = 1,1474
Epoch 416, Update 2500, Cost = 1,1164
Epoch 433, Update 2600, Cost = 1,2900
Epoch 449, Update 2700, Cost = 1,1473
Epoch 466, Update 2800, Cost = 1,1164
Epoch 483, Update 2900, Cost = 1,2899
Epoch 499, Update 3000, Cost = 1,1473
Epoch 516, Update 3100, Cost = 1,1164
Epoch 533, Update 3200, Cost = 1,2899
Epoch 549, Update 3300, Cost = 1,1473
Epoch 566, Update 3400, Cost = 1,1164
Epoch 583, Update 3500, Cost = 1,2899
Epoch 599, Update 3600, Cost = 1,1473
Epoch 616, Update 3700, Cost = 1,1164
Epoch 633, Update 3800, Cost = 1,2899
Epoch 649, Update 3900, Cost = 1,1473
Epoch 666, Update 4000, Cost = 1,1164
Epoch 683, Update 4100, Cost = 1,2899
Epoch 699, Update 4200, Cost = 1,1473

Translations:
<s> Che ore sono ? </s>
<s> Questa è la mia casa </s>
<s> Ti [CLS] <s> stanco </s>
             */

            /*
             Epoch 16, Update 100, Cost = 5,4185
Epoch 33, Update 200, Cost = 3,3528
Epoch 49, Update 300, Cost = 2,0124
Epoch 66, Update 400, Cost = 1,5609
Epoch 83, Update 500, Cost = 1,5046
Epoch 99, Update 600, Cost = 1,1694
Epoch 116, Update 700, Cost = 1,1006
Epoch 133, Update 800, Cost = 1,1821
Epoch 149, Update 900, Cost = 0,9881
Epoch 166, Update 1000, Cost = 0,9873
Epoch 183, Update 1100, Cost = 1,0962
Epoch 199, Update 1200, Cost = 0,9379
Epoch 216, Update 1300, Cost = 0,9552
Epoch 233, Update 1400, Cost = 1,0715
Epoch 249, Update 1500, Cost = 0,9232
Epoch 266, Update 1600, Cost = 0,9457
Epoch 283, Update 1700, Cost = 1,0639
Epoch 299, Update 1800, Cost = 0,9187
Epoch 316, Update 1900, Cost = 0,9427
Epoch 333, Update 2000, Cost = 1,0616
Epoch 349, Update 2100, Cost = 0,9173
Epoch 366, Update 2200, Cost = 0,9418
Epoch 383, Update 2300, Cost = 1,0609
Epoch 399, Update 2400, Cost = 0,9169
Epoch 416, Update 2500, Cost = 0,9415
Epoch 433, Update 2600, Cost = 1,0607
Epoch 449, Update 2700, Cost = 0,9167
Epoch 466, Update 2800, Cost = 0,9414
Epoch 483, Update 2900, Cost = 1,0606
Epoch 499, Update 3000, Cost = 0,9167
Epoch 516, Update 3100, Cost = 0,9414
Epoch 533, Update 3200, Cost = 1,0606
Epoch 549, Update 3300, Cost = 0,9167
Epoch 566, Update 3400, Cost = 0,9414
Epoch 583, Update 3500, Cost = 1,0606
Epoch 599, Update 3600, Cost = 0,9167
Epoch 616, Update 3700, Cost = 0,9414
Epoch 633, Update 3800, Cost = 1,0606
Epoch 649, Update 3900, Cost = 0,9167
Epoch 666, Update 4000, Cost = 0,9414
Epoch 683, Update 4100, Cost = 1,0606
Epoch 699, Update 4200, Cost = 0,9167

Translations:
<s> Che ore sono ? </s>
<s> Questa è la </s>
<s> è amo Questo Questo Questo </s>
             */

            /*
             Epoch 16, Update 100, Cost = 5,6821
Epoch 33, Update 200, Cost = 3,0756
Epoch 49, Update 300, Cost = 2,1387
Epoch 66, Update 400, Cost = 1,6491
Epoch 83, Update 500, Cost = 1,4629
Epoch 99, Update 600, Cost = 1,2984
Epoch 116, Update 700, Cost = 1,1957
Epoch 133, Update 800, Cost = 1,1771
Epoch 149, Update 900, Cost = 1,1151
Epoch 166, Update 1000, Cost = 1,0791
Epoch 183, Update 1100, Cost = 1,0977
Epoch 199, Update 1200, Cost = 1,0621
Epoch 216, Update 1300, Cost = 1,0448
Epoch 233, Update 1400, Cost = 1,0740
Epoch 249, Update 1500, Cost = 1,0463
Epoch 266, Update 1600, Cost = 1,0345
Epoch 283, Update 1700, Cost = 1,0668
Epoch 299, Update 1800, Cost = 1,0415
Epoch 316, Update 1900, Cost = 1,0313
Epoch 333, Update 2000, Cost = 1,0646
Epoch 349, Update 2100, Cost = 1,0400
Epoch 366, Update 2200, Cost = 1,0303
Epoch 383, Update 2300, Cost = 1,0639
Epoch 399, Update 2400, Cost = 1,0395
Epoch 416, Update 2500, Cost = 1,0300
Epoch 433, Update 2600, Cost = 1,0637
Epoch 449, Update 2700, Cost = 1,0394
Epoch 466, Update 2800, Cost = 1,0299
Epoch 483, Update 2900, Cost = 1,0637
Epoch 499, Update 3000, Cost = 1,0394
Epoch 516, Update 3100, Cost = 1,0299
Epoch 533, Update 3200, Cost = 1,0637
Epoch 549, Update 3300, Cost = 1,0393
Epoch 566, Update 3400, Cost = 1,0299
Epoch 583, Update 3500, Cost = 1,0637
Epoch 599, Update 3600, Cost = 1,0393
Epoch 616, Update 3700, Cost = 1,0299
Epoch 633, Update 3800, Cost = 1,0637
Epoch 649, Update 3900, Cost = 1,0393
Epoch 666, Update 4000, Cost = 1,0299
Epoch 683, Update 4100, Cost = 1,0637
Epoch 699, Update 4200, Cost = 1,0393

Translations:
<s> Che ore sono ? </s>
<s> Questa è la la la </s>
<s> Questa è è </s>
             */

            /*
             Epoch 16, Update 100, Cost = 5,7998
          Epoch 33, Update 200, Cost = 3,2184
          Epoch 49, Update 300, Cost = 2,2065
          Epoch 66, Update 400, Cost = 1,7045
          Epoch 83, Update 500, Cost = 1,5789
          Epoch 99, Update 600, Cost = 1,3602
          Epoch 116, Update 700, Cost = 1,2257
          Epoch 133, Update 800, Cost = 1,2841
          Epoch 149, Update 900, Cost = 1,1718
          Epoch 166, Update 1000, Cost = 1,1075
          Epoch 183, Update 1100, Cost = 1,2056
          Epoch 199, Update 1200, Cost = 1,1201
          Epoch 216, Update 1300, Cost = 1,0743
          Epoch 233, Update 1400, Cost = 1,1829
          Epoch 249, Update 1500, Cost = 1,1049
          Epoch 266, Update 1600, Cost = 1,0643
          Epoch 283, Update 1700, Cost = 1,1761
          Epoch 299, Update 1800, Cost = 1,1002
          Epoch 316, Update 1900, Cost = 1,0613
          Epoch 333, Update 2000, Cost = 1,1739
          Epoch 349, Update 2100, Cost = 1,0988
          Epoch 366, Update 2200, Cost = 1,0603
          Epoch 383, Update 2300, Cost = 1,1732
          Epoch 399, Update 2400, Cost = 1,0983
          Epoch 416, Update 2500, Cost = 1,0600
          Epoch 433, Update 2600, Cost = 1,1730
          Epoch 449, Update 2700, Cost = 1,0982
          Epoch 466, Update 2800, Cost = 1,0599
          Epoch 483, Update 2900, Cost = 1,1730
          Epoch 499, Update 3000, Cost = 1,0982
          Epoch 516, Update 3100, Cost = 1,0599
          Epoch 533, Update 3200, Cost = 1,1730
          Epoch 549, Update 3300, Cost = 1,0982
          Epoch 566, Update 3400, Cost = 1,0599
          Epoch 583, Update 3500, Cost = 1,1730
          Epoch 599, Update 3600, Cost = 1,0982
          Epoch 616, Update 3700, Cost = 1,0599
          Epoch 633, Update 3800, Cost = 1,1730
          Epoch 649, Update 3900, Cost = 1,0982
          Epoch 666, Update 4000, Cost = 1,0599
          Epoch 683, Update 4100, Cost = 1,1730
          Epoch 699, Update 4200, Cost = 1,0982

          Translations:
          <s> Che ore sono ? </s>
          <s> Questa è la mia </s>
          <s> Parlo Questo Questo libro </s>
             */

            Console.ReadLine();
        }
    }
}
