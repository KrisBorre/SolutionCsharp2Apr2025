using Seq2SeqSharp;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Enums;
using Seq2SeqSharp.LearningRate;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;

// ChatGPT, aangepast
namespace ConsoleDutchEnglishSeq2Seq1Jun2025
{
    internal class Program
    {
        static void Main(string[] args)
        {
            string srcLang = "NL";
            string tgtLang = "EN";

            var trainData = new List<(string src, string tgt)>
            {
                ("Hallo , hoe gaat het ?", "Hello , how are you ?"),
                ("Ik hou van jou", "I love you"),
                ("Dit is een boek", "This is a book"),
                ("Zij is een lerares", "She is a teacher"),
                ("Wat is jouw naam ?", "What is your name ?"),
                ("Dit is mijn huis", "This is my house"),
                ("Ik spreek een beetje Engels", "I speak a little English"),
                ("Waar is het station ?", "Where is the station ?"),
                ("Hoe laat is het ?", "What time is it ?"),
                ("Ik ben moe", "I am tired")
            };

            string srcTrainFile = "train.nl.snt";
            string tgtTrainFile = "train.en.snt";

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
                MaxEpochNum = 100,
                MaxTokenSizePerBatch = 10,
                ValMaxTokenSizePerBatch = 10,
                MaxSrcSentLength = 20,
                MaxTgtSentLength = 20,
                PaddingType = PaddingEnums.AllowPadding,
                TooLongSequence = TooLongSequence.Ignore,
                ProcessorType = ProcessorTypeEnums.CPU,
                ModelFilePath = "nl2en_epoch100.model",
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
            opts.ModelFilePath = "nl2en_epoch100.model.trained";
            var inferModel = new Seq2Seq(opts);

            string testInputPath = "test_input.nl.snt";
            string testOutputPath = "test_output.en.snt";
            File.WriteAllLines(testInputPath, new[]
            {
                "Hallo , hoe gaat het ?",
                "Waar is het station ?",
                "Ik hou van jou",
                "Zij is een lerares",
                "Ik ben moe"
            });

            inferModel.Test(
                inputTestFile: testInputPath,
                outputFile: testOutputPath,
                batchSize: 1,
                decodingOptions: opts.CreateDecodingOptions(),
                srcSpmPath: null,
                tgtSpmPath: null);

            Console.WriteLine("\nTranslations:");
            foreach (var line in File.ReadLines(testOutputPath))
            {
                Console.WriteLine(line);
            }

            /*
             Epoch 14, Update 100, Cost = 6,9185
Epoch 28, Update 200, Cost = 3,6419
Epoch 42, Update 300, Cost = 2,8462
Epoch 57, Update 400, Cost = 3,7484
Epoch 71, Update 500, Cost = 1,9495
Epoch 85, Update 600, Cost = 1,7472
Epoch 99, Update 700, Cost = 1,7218

Translations:
<s> Hello , how are you ? </s>
<s> Where is the station ? </s>
<s> I is a </s>
<s> She is a </s>
<s> I am tired </s>
             */


            /*
             Epoch 14, Update 100, Cost = 6,4327
Epoch 28, Update 200, Cost = 3,5246
Epoch 42, Update 300, Cost = 2,8790
Epoch 57, Update 400, Cost = 3,8673
Epoch 71, Update 500, Cost = 1,8907
Epoch 85, Update 600, Cost = 1,7920
Epoch 99, Update 700, Cost = 1,6374

Translations:
<s> Hello , how are you ? </s>
<s> Where is the station ? </s>
<s> I love </s>
<s> She is a teacher </s>
<s> I am tired </s>
             */

            Console.ReadLine();
        }
    }
}
