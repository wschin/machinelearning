// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Runtime.Model.Pmf
{
    /// <summary>
    /// A context for defining a ONNX output.
    /// </summary>
    internal sealed class PmfContextImpl : PmfContext
    {
        private readonly Dictionary<string, string> _columnNameMap;
        // All existing variable names. New variables must not exist in this set.
        private readonly HashSet<string> _variableNames;
        // All existing node names. New node names must not alrady exist in this set.
        private readonly HashSet<string> _nodeNames;
 
        public override string AddIntermediateVariable(ColumnType type, string colName, bool skip = false)
        {
            return "None";
        }

        public override bool ContainsColumn(string colName)
        {
            return true;
        }

        public override string GetOperatorName(string prefix)
        {
            return "None";
        }

        public override string GetVariableName(string colName)
        {
            return "None";
        }

        internal void AddInputVariable(ColumnType columnType, string colName)
        {
        }

        internal object TryGetVariableName(string idataviewColumnName)
        {
            return "None";
        }

        internal void AddOutputVariable(ColumnType columnType, object variableName)
        {
        }
    }
}
