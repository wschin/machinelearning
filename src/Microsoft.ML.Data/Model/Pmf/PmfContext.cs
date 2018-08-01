// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Runtime.Model.Pmf
{
    /// <summary>
    /// A context for defining a ONNX output. The context internally contains the model-in-progress being built. This
    /// same context object is iteratively given to exportable components via the <see cref="ICanSavePmf"/> interface
    /// and subinterfaces, that attempt to express their operations as ONNX nodes, if they can. At the point that it is
    /// given to a component, all other components up to that component have already attempted to express themselves in
    /// this context, with their outputs possibly available in the ONNX graph.
    /// </summary>
    public abstract class PmfContext
    {
        /// <summary>
        /// Generates a unique name for the node based on a prefix.
        /// </summary>
        /// <param name="prefix">The prefix for the node</param>
        /// <returns>A name that has not yet been returned from this function, starting with <paramref name="prefix"/></returns>
        public abstract string GetOperatorName(string prefix);

        /// <summary>
        /// Looks up whether a given data view column has a mapping in the ONNX context. Once confirmed, callers can
        /// safely call <see cref="GetVariableName(string)"/>.
        /// </summary>
        /// <param name="colName">The data view column name</param>
        /// <returns>Whether the column is mapped in this context</returns>
        public abstract bool ContainsColumn(string colName);

        /// <summary>
        /// ONNX variables are referred to by name. At each stage of a ML.NET pipeline, the corresponding
        /// <see cref="IDataView"/>'s column names will map to a variable in the ONNX graph if the intermediate steps
        /// used to calculate that value are things we knew how to save as ONNX. Retrieves the variable name that maps
        /// to the <see cref="IDataView"/> column name at a given point in the pipeline execution. Callers should
        /// probably confirm with <see cref="ContainsColumn(string)"/> whether a mapping for that data view column
        /// already exists.
        /// </summary>
        /// <param name="colName">The data view column name</param>
        /// <returns>The ONNX variable name corresponding to that data view column</returns>
        public abstract string GetVariableName(string colName);

        /// <summary>
        /// Establishes a new mapping from an data view column in the context, if necessary generates a unique name, and
        /// returns that newly allocated name.
        /// </summary>
        /// <param name="type">The data view type associated with this column name</param>
        /// <param name="colName">The data view column name</param>
        /// <param name="skip">Whether we should skip the process of establishing the mapping from data view column to
        /// ONNX variable name.</param>
        /// <returns>The returned value is the name of the variable corresponding </returns>
        public abstract string AddIntermediateVariable(ColumnType type, string colName, bool skip = false);
    }
}