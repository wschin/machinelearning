// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using Google.Protobuf;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Command;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model.Pmf;
using Newtonsoft.Json;

[assembly: LoadableClass(SavePmfCommand.Summary, typeof(SavePmfCommand), typeof(SavePmfCommand.Arguments), typeof(SignatureCommand),
    "Save PMF", "SavePmf", DocName = "command/SavePmf.md")]

[assembly: LoadableClass(typeof(void), typeof(SavePmfCommand), null, typeof(SignatureEntryPointModule), "SavePmf")]


namespace Microsoft.ML.Runtime.Model.Pmf
{
    public sealed class SavePmfCommand : DataCommand.ImplBase<SavePmfCommand.Arguments>
    {
        public const string Summary = "Given a data model, write out the corresponding PMF (protable model format).";
        public const string LoadName = "SavePmf";

        public sealed class Arguments : DataCommand.ArgumentsBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "The path to write the output PMF to.", SortOrder = 1)]
            public string Pmf;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The path to write the output JSON to.", SortOrder = 2)]
            public string Json;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The 'name' property in the output ONNX. By default this will be the PMF extension-less name.", NullName = "<Auto>", SortOrder = 3)]
            public string Name;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The 'domain' property in the output ONNX.", NullName = "<Auto>", SortOrder = 4)]
            public string Domain;

            [Argument(ArgumentType.AtMostOnce, Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly, HelpText = "Comma delimited list of input column names to drop", ShortName = "idrop", SortOrder = 5)]
            public string InputsToDrop;

            [Argument(ArgumentType.AtMostOnce, Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly, HelpText = "Comma delimited list of output column names to drop", ShortName = "odrop", SortOrder = 7)]
            public string OutputsToDrop;

            [Argument(ArgumentType.AtMostOnce, Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly, HelpText = "Whether we should attempt to load the predictor and attach the scorer to the pipeline if one is present.", ShortName = "pred", SortOrder = 9)]
            public bool? LoadPredictor;

            [Argument(ArgumentType.Required, Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly, HelpText = "Model that needs to be converted to Pmf format.", SortOrder = 10)]
            public ITransformModel Model;
        }

        // PMF model file (in protobuf format) name (i.e., the path to PMF model file we are going to produce)
        private readonly string _outputModelPath;
        // PMF model file (in text format) name (i.e., the path to PMF model file we are going to produce)
        private readonly string _outputJsonModelPath;
        // PMF model name (can be different than model file name)
        private readonly string _name;
        private readonly string _domain;
        private readonly bool? _loadPredictor;
        private readonly HashSet<string> _inputsToDrop;
        private readonly HashSet<string> _outputsToDrop;
        private readonly ITransformModel _model;
        private const string ProducerName = "ML.NET";
        private const long ModelVersion = 0;

        public SavePmfCommand(IHostEnvironment env, Arguments args)
                : base(env, args, LoadName)
        {
            Host.CheckValue(args, nameof(args));
            Utils.CheckOptionalUserDirectory(args.Pmf, nameof(args.Json));
            _outputModelPath = string.IsNullOrWhiteSpace(args.Pmf) ? null : args.Pmf;
            _outputJsonModelPath = string.IsNullOrWhiteSpace(args.Json) ? null : args.Json;
            if (args.Name == null && _outputModelPath != null)
                _name = Path.GetFileNameWithoutExtension(_outputModelPath);
            else if (!string.IsNullOrWhiteSpace(args.Name))
                _name = args.Name;

            _loadPredictor = args.LoadPredictor;
            _inputsToDrop = CreateDropMap(args.InputsToDrop?.Split(','));
            _outputsToDrop = CreateDropMap(args.OutputsToDrop?.Split(','));
            _domain = args.Domain;
            _model = args.Model;
        }

        private static HashSet<string> CreateDropMap(string[] toDrop)
        {
            if (toDrop == null)
                return new HashSet<string>();

            return new HashSet<string>(toDrop);
        }

        public override void Run()
        {
            using (var ch = Host.Start("Run"))
            {
                Run(ch);
                ch.Done();
            }
        }

        private void Run(IChannel ch)
        {
            IDataLoader loader = null;
            IPredictor rawPred = null;
            IDataView view;
            RoleMappedSchema trainSchema = null;

            // What commands can invoke this? What does _model==null mean? Fail to load model? model file not specified?
            if (_model == null)
            {
                // What commands can invoke this?
                if (string.IsNullOrEmpty(Args.InputModelFile))
                {
                    loader = CreateLoader();
                    rawPred = null;
                    trainSchema = null;
                    Host.CheckUserArg(Args.LoadPredictor != true, nameof(Args.LoadPredictor),
                        "Cannot be set to true unless " + nameof(Args.InputModelFile) + " is also specifified.");
                }
                else
                {
                    // What commands can invoke this?
                    LoadModelObjects(ch, _loadPredictor, out rawPred, true, out trainSchema, out loader);
                }
                view = loader;
            }
            else
            {
                // What commands can invoke this?
                view = _model.Apply(Host, new EmptyDataView(Host, _model.InputSchema));
            }

            // Get the transform chain.
            IDataView source;
            IDataView end;
            LinkedList<ITransformCanSavePmf> transforms;
            GetPipe(ch, view, out source, out end, out transforms);
            Host.Assert(transforms.Count == 0 || transforms.Last.Value == end);

            var assembly = System.Reflection.Assembly.GetExecutingAssembly();
            var versionInfo = System.Diagnostics.FileVersionInfo.GetVersionInfo(assembly.Location);

            var ctx = new PmfContext();
            // If we have a predictor, try to get the scorer for it.
            if (rawPred != null)
            {
                RoleMappedData data;
                if (trainSchema != null)
                    data = new RoleMappedData(end, trainSchema.GetColumnRoleNames());
                else
                {
                    // We had a predictor, but no roles stored in the model. Just suppose
                    // default column names are OK, if present.
                    data = new RoleMappedData(end, DefaultColumnNames.Label,
                        DefaultColumnNames.Features, DefaultColumnNames.GroupId, DefaultColumnNames.Weight, DefaultColumnNames.Name, opt: true);
                }

                var scorePipe = ScoreUtils.GetScorer(rawPred, data, Host, trainSchema);
                var scoreOnnx = scorePipe as ITransformCanSavePmf;
                if (scoreOnnx?.CanSavePmf == true)
                {
                    Host.Assert(scorePipe.Source == end);
                    end = scorePipe;
                    transforms.AddLast(scoreOnnx);
                }
                else
                {
                    Contracts.CheckUserArg(_loadPredictor != true,
                        nameof(Arguments.LoadPredictor), "We were explicitly told to load the predictor but we do not know how to save it as ONNX.");
                    ch.Warning("We do not know how to save the predictor as ONNX. Ignoring.");
                }
            }
            else
            {
                Contracts.CheckUserArg(_loadPredictor != true,
                    nameof(Arguments.LoadPredictor), "We were explicitly told to load the predictor but one was not present.");
            }

            HashSet<string> inputColumns = new HashSet<string>();
            //Create graph inputs.
            for (int i = 0; i < source.Schema.ColumnCount; i++)
            {
                string colName = source.Schema.GetColumnName(i);
                if(_inputsToDrop.Contains(colName))
                    continue;

                ctx.AddInputVariable(source.Schema.GetColumnType(i), colName);
                inputColumns.Add(colName);
            }

            //Create graph nodes, outputs and intermediate values.
            foreach (var trans in transforms)
            {
                Host.Assert(trans.CanSavePmf);
                trans.SaveAsPmf(ctx);
            }

            //Add graph outputs.
            for (int i = 0; i < end.Schema.ColumnCount; ++i)
            {
                if (end.Schema.IsHidden(i))
                    continue;

                var idataviewColumnName = end.Schema.GetColumnName(i);;
                if (_outputsToDrop.Contains(idataviewColumnName) || _inputsToDrop.Contains(idataviewColumnName))
                    continue;

                var variableName = ctx.RetrieveVariableNameOrCreateOne(idataviewColumnName);
                if (variableName != null)
                    ctx.AddOutputVariable(end.Schema.GetColumnType(i), variableName);
            }
            var model = ctx.MakeModel();
            if (_outputModelPath != null)
            {
                using (var file = Host.CreateOutputFile(_outputModelPath))
                using (var stream = file.CreateWriteStream())
                    model.WriteTo(stream);
            }

            if (_outputJsonModelPath != null)
            {
                using (var file = Host.CreateOutputFile(_outputJsonModelPath))
                using (var stream = file.CreateWriteStream())
                using (var writer = new StreamWriter(stream))
                {
                    var parsedJson = JsonConvert.DeserializeObject(model.ToString());
                    writer.Write(JsonConvert.SerializeObject(parsedJson, Formatting.Indented));
                }
            }
        }

        private void GetPipe(IChannel ch, IDataView end, out IDataView source, out IDataView trueEnd, out LinkedList<ITransformCanSavePmf> transforms)
        {
            Host.AssertValue(end);
            source = trueEnd = (end as CompositeDataLoader)?.View ?? end;
            IDataTransform transform = source as IDataTransform;
            transforms = new LinkedList<ITransformCanSavePmf>();
            while (transform != null)
            {
                ITransformCanSavePmf pmfTransform = transform as ITransformCanSavePmf;
                if (pmfTransform == null || !pmfTransform.CanSavePmf)
                {
                    ch.Warning("Had to stop walkback of pipeline at {0} since it cannot save itself as ONNX.", transform.GetType().Name);
                    while (source as IDataTransform != null)
                        source = (source as IDataTransform).Source;

                    return;
                }
                transforms.AddFirst(pmfTransform);
                transform = (source = transform.Source) as IDataTransform;
            }

            Host.AssertValue(source);
        }

        public sealed class Output
        {
        }

        [TlcModule.EntryPoint(Name = "Models.PmfConverter", Desc = "Converts the model to PMF format.", UserName = "PMF Converter.")]
        public static Output Apply(IHostEnvironment env, Arguments input)
        {
            new SavePmfCommand(env, input).Run();
            return new Output();
        }
    }
}
