// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Runtime.Model.Pmf
{
    public interface ICanSavePmf
    {
        bool CanSavePmf { get; }
    }

    public interface ITransformCanSavePmf: ICanSavePmf, IDataTransform
    {
        void SaveAsPmf(PmfContext ctx);
    }

    public interface IBindableCanSavePmf : ICanSavePmf, ISchemaBindableMapper
    {
        bool SaveAsPmf(PmfContext ctx, RoleMappedSchema schema, string[] outputNames);
    }

    public interface ISingleCanSavePmf : ICanSavePmf
    {
        bool SaveAsPmf(PmfContext ctx, string[] outputNames, string featureColumn);
    }

    public interface IDistCanSavePmf : ISingleCanSavePmf, IValueMapperDist
    {
        new bool SaveAsPmf(PmfContext ctx, string[] outputNames, string featureColumn);
    }
}