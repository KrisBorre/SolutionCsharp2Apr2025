using Seq2SeqSharp;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Enums;
using Seq2SeqSharp.LearningRate;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;

// ChatGPT, modified
namespace ConsoleDutchItalianSeq2Seq07Jul2025
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
                HiddenSize = 512,
                SrcEmbeddingDim = 512,
                TgtEmbeddingDim = 512,
                MaxEpochNum = 200,
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
 Epoch 16, Update 100, Cost = 0,6977
Epoch 33, Update 200, Cost = 0,0510
Epoch 49, Update 300, Cost = 0,0218
Epoch 66, Update 400, Cost = 0,0111
Epoch 83, Update 500, Cost = 0,0065
Epoch 99, Update 600, Cost = 0,0036
Epoch 116, Update 700, Cost = 0,0021
Epoch 133, Update 800, Cost = 0,0014
Epoch 149, Update 900, Cost = 0,0009
Epoch 166, Update 1000, Cost = 1,0608
Epoch 183, Update 1100, Cost = 0,4263
Epoch 199, Update 1200, Cost = 0,0907

Translations:
<s> Che ore sono ? </s>
<s> Questa è la mia casa </s>
<s> Lei è la mia </s>
            */

            /*
             Epoch 16, Update 100, Cost = 0,4936
Epoch 33, Update 200, Cost = 0,0647
Epoch 49, Update 300, Cost = 0,0206
Epoch 66, Update 400, Cost = 0,0099
Epoch 83, Update 500, Cost = 0,0060
Epoch 99, Update 600, Cost = 0,0031
Epoch 116, Update 700, Cost = 0,0018
Epoch 133, Update 800, Cost = 0,0012
Epoch 149, Update 900, Cost = 0,0008
Epoch 166, Update 1000, Cost = 1,2624
Epoch 183, Update 1100, Cost = 0,5307
Epoch 199, Update 1200, Cost = 0,1099

Translations:
<s> Che ore sono ? </s>
<s> Questa è la mia casa </s>
<s> Sono è la </s>
             */


            /*
             Epoch 16, Update 100, Cost = 0,6372
Epoch 33, Update 200, Cost = 0,0578
Epoch 49, Update 300, Cost = 0,0228
Epoch 66, Update 400, Cost = 0,0116
Epoch 83, Update 500, Cost = 0,0074
Epoch 99, Update 600, Cost = 0,0039
Epoch 116, Update 700, Cost = 0,0023
Epoch 133, Update 800, Cost = 0,0016
Epoch 149, Update 900, Cost = 0,0010
Epoch 166, Update 1000, Cost = 1,8711
Epoch 183, Update 1100, Cost = 0,5541
Epoch 199, Update 1200, Cost = 0,0959

Translations:
<s> Che ore sono ? </s>
<s> Questa è la mia casa </s>
<s> Ti amo </s>
             */
            Console.ReadLine();
        }
    }
}
