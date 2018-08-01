// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Runtime.Model.Pmf
{
    public interface ICanSavePmf
    {
        /// <summary>
        /// Whether this object really is capable of saving itself as part of an ONNX
        /// pipeline. An implementor of this object might implement this interface,
        /// but still return <c>false</c> if there is some characteristic of this object
        /// only detectable during runtime that would prevent its being savable. (E.g.,
        /// it may wrap some other object that may or may not be savable.)
        /// </summary>
        bool CanSavePmf { get; }
    }

    /// <summary>
    /// This data model component is savable as ONNX.
    /// </summary>
    public interface ITransformCanSavePmf: ICanSavePmf, IDataTransform
    {
        /// <summary>
        /// Create the associated transform in PMF format
        /// </summary>
        /// <param name="ctx">The ONNX program being built</param>
        void SaveAsPmf(PmfContext ctx);
    }

    /// <summary>
    /// This <see cref="ISchemaBindableMapper"/> is savable in ONNX. Note that this is
    /// typically called within an <see cref="IDataScorerTransform"/> that is wrapping
    /// this mapper, and has already been bound to it.
    /// </summary>
    public interface IBindableCanSavePmf : ICanSavePmf, ISchemaBindableMapper
    {
        /// <summary>
        /// Save as ONNX. If <see cref="ICanSavePmf.CanSavePmf"/> is
        /// <c>false</c> this should not be called. This method is intended to be called
        /// by the wrapping scorer transform, and is intended to produce enough information
        /// for that purpose.
        /// </summary>
        /// <param name="ctx">The ONNX program being built</param>
        /// <param name="schema">The role mappings that was passed to this bindable
        /// object, when the <see cref="ISchemaBoundMapper"/> was created that this transform
        /// is wrapping</param>
        /// <param name="outputNames">Since this method is called from a scorer transform,
        /// it is that transform that controls what the output column names will be, of
        /// the outputs produced by this bindable mapper. This is the array that holds
        /// those names, so that implementors of this method know what to produce in
        /// <paramref name="ctx"/>.</param>
        bool SaveAsPmf(PmfContext ctx, RoleMappedSchema schema, string[] outputNames);
    }

    /// <summary>
    /// For simple mappers. Intended to be used for <see cref="IValueMapper"/> and
    /// <see cref="Microsoft.ML.Runtime.Internal.Calibration.ICalibrator"/> instances.
    /// </summary>
    public interface ISingleCanSavePmf : ICanSavePmf
    {
        bool SaveAsPmf(PmfContext ctx, string[] outputNames, string featureColumn);
    }

    /// <summary>
    /// For simple mappers. Intended to be used for <see cref="IValueMapperDist"/>
    /// instances.
    /// </summary>
    public interface IDistCanSavePmf : ISingleCanSavePmf, IValueMapperDist
    {
        new bool SaveAsPmf(PmfContext ctx, string[] outputNames, string featureColumn);
    }
}
