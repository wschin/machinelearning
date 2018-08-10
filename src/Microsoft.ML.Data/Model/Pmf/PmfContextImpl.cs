// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using LotusvNext;
using LotusvNext.Expressions;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Runtime.Model.Pmf
{
    /// <summary>
    /// A context for defining a ONNX output.
    /// </summary>
    internal sealed class PmfContextImpl : PmfContext
    {
        private Dictionary<string, string> _ourColumnNameToPmfNameMap;
        private Dictionary<string, string> _ourOperatorNameToPmfNameMap;
        // All existing variable names. New variables must not exist in this set.
        private HashSet<string> _variableNamePool;
        // All existing node names. New node names must not alrady exist in this set.
        private HashSet<string> _operatorNamePool;
        private Dictionary<string, LotusvNext.Types.TypeProto> _modelInputParameters;
        private Dictionary<string, LotusvNext.Types.TypeProto> _modelOutputParameters;
        private List<LotusvNext.Expressions.Expression> _expressions;

        public PmfContextImpl()
        {
            _variableNamePool = new HashSet<string>();
            _operatorNamePool = new HashSet<string>();
            _ourColumnNameToPmfNameMap = new Dictionary<string, string>();
            _ourOperatorNameToPmfNameMap = new Dictionary<string, string>();
            _modelInputParameters = new Dictionary<string, LotusvNext.Types.TypeProto>();
            _modelOutputParameters = new Dictionary<string, LotusvNext.Types.TypeProto>();
            _expressions = new List<LotusvNext.Expressions.Expression>();
        }

        private string CreateUniqueString(ref HashSet<string> pool, string prefix)
        {
            if (!pool.Contains(prefix))
            {
                pool.Add(prefix);
                return prefix;
            }

            // The string pool contains the input prefix, so we are going to append something to make a unique string.
 
            // The index will be appended
            int count = 1;
            // Candidate string
            string derivedString = prefix + count;;
            // Keep increasing count until prefix + count cannot be found in the pool
            while(pool.Contains(derivedString))
            {
                derivedString = prefix + ++count;
            }
            // Add derived_string into the pool so that we will not declare the same string again in the future
            pool.Add(derivedString);
            return derivedString;
        }
        public override string CreateOperatorName(string prefix)
        {
            string name = CreateUniqueString(ref _variableNamePool, prefix);
            _ourOperatorNameToPmfNameMap[prefix] = name;
            return name;
        }

        public override string CreateVariableName(string prefix)
        {
            string name = CreateUniqueString(ref _operatorNamePool, prefix);
            _ourColumnNameToPmfNameMap[prefix] = name;
            return name;
        }

        public override string RetrieveVariableNameOrCreateOne(string prefix)
        {
            if (_ourColumnNameToPmfNameMap.ContainsKey(prefix))
                return _ourColumnNameToPmfNameMap[prefix];
            else
                return CreateVariableName(prefix);
        }

        // Use to create I/O for FunctionalModelProto
        public override void AddInputVariable(ColumnType columnType, string colName)
        {
            // Declare name in IR
            var name = CreateVariableName(colName);
            // Create TypeProto according to the input column
            var typeProto = PmfUtils.TranslateColumnTypeToTypeProto(columnType);
            // Create ParameterDeclProto based on information above
            _modelInputParameters[name] = typeProto;
        }

        public override void AddOutputVariable(ColumnType columnType, string colName)
        {
            // Declare name in IR
            var name = CreateVariableName(colName);
            // Create TypeProto according to the input column
            var typeProto = PmfUtils.TranslateColumnTypeToTypeProto(columnType);
            _modelOutputParameters[name] = typeProto;
        }
        public override void AddExpression(LotusvNext.Expressions.Expression expression)
        {
            _expressions.Add(expression);
        }
        public override FunctionalModelProto MakeFunctionalModel()
        {
            var inputs = new List<LotusvNext.Types.TypeProto.Types.ParameterDeclProto>();
            var outputs = new List<LotusvNext.Types.TypeProto.Types.ParameterDeclProto>();
            foreach (var i in _modelInputParameters)
                inputs.Add(PmfUtils.MakeParameterDeclProtoTensor(i.Key, i.Value));
            foreach (var i in _modelOutputParameters)
                outputs.Add(PmfUtils.MakeParameterDeclProtoTensor(i.Key, i.Value));

            return PmfUtils.MakeFunctionalModelProto("simple", inputs, outputs, _expressions);
        }
        public override ModelProto MakeModel()
        {
            var modelProto = new ModelProto
            {
                IrVersion = 0L,
                ProducerName = "ML.NET",
                ProducerVersion = "0",
                Domain = "com.microsoft",
                ModelVersion = 0L,
                Model = MakeFunctionalModel()
            };
            return modelProto;
        }
    }
}
