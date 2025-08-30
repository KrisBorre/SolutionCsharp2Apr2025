using Seq2SeqSharp;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Enums;
using Seq2SeqSharp.LearningRate;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;

// ChatGPT, aangepast
namespace ConsoleDutchItalianSeq2Seq1Jun2025
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
                MaxEpochNum = 200, // 60,
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
                "Hallo , hoe gaat het ?",
                "Waar is het station ?",
                "Ik hou van jou"
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
            Epoch 16, Update 100, Cost = 5,4934
Epoch 33, Update 200, Cost = 3,4264
Epoch 49, Update 300, Cost = 2,1112

Translations:
<s> Ciao , come stai ? </s>
<s> Dove è la stazione ? </s>
<s> Ti amo </s>

            */

            /*
             Epoch 16, Update 100, Cost = 6,1310
Epoch 33, Update 200, Cost = 3,3916
Epoch 49, Update 300, Cost = 2,4591
Epoch 66, Update 400, Cost = 1,8864
Epoch 83, Update 500, Cost = 1,6897
Epoch 99, Update 600, Cost = 1,5064
Epoch 116, Update 700, Cost = 1,3583
Epoch 133, Update 800, Cost = 1,3461
Epoch 149, Update 900, Cost = 1,2810
Epoch 166, Update 1000, Cost = 1,2234
Epoch 183, Update 1100, Cost = 1,2553
Epoch 199, Update 1200, Cost = 1,2189

Translations:
<s> Ciao , come stai ? </s>
<s> Dove è la stazione ? </s>
<s> </s>

             */

            Console.ReadLine();
        }
    }
}
