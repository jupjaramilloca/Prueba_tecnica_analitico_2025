-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-- Equipo Vicepresidencia de Ecosistemas
-----------------------------------------------------------------------------
-- Fecha Creación: 20250203
-- Última Fecha Modificación: 20250203
-- Autores: juajaram
-- Últimos Autores: juajaram
-- Descripción:     Cambiar la descripción del SQL
-----------------------------------------------------------------------------
---------------------------------- INSUMOS ----------------------------------
-- {in_oozie}
--------------------------------- RESULTADOS --------------------------------
-- {zona_p}.{prefijo}temporal_ads_gpkg
-----------------------------------------------------------------------------
-------------------------------- Query Start --------------------------------


drop table if exists proceso_ecosistemas.opeco_prediccion_pagos purge;


create table if not exists proceso_ecosistemas.opeco_prediccion_pagos stored as parquet tblproperties ('transactional'='false') as
    with prediction as (
        select 
        concat(
            cast(nit_enmascarado as string),
            "#",
            cast(num_oblig_orig_enmascarado as string),
            "#",
            cast(num_oblig_enmascarado as string)) as ID,
        predictions as var_rpta_alt
        from proceso_ecosistemas.opeco_sabana_test_pred

    ) 

    select *
    from prediction

;


drop table if exists proceso_ecosistemas.opeco_prediccion_pagos_resultado purge;


create table if not exists proceso_ecosistemas.opeco_prediccion_pagos_resultado stored as parquet tblproperties ('transactional'='false') as

    with prediction as (



        select 
        concat(
            cast(nit_enmascarado as string),
            "#",
            cast(num_oblig_orig_enmascarado as string),
            "#",
            cast(num_oblig_enmascarado as string)) as ID,
        predictions as var_rpta_alt,
        cast(probability as decimal(5,5)) as Prob_uno
        from proceso_ecosistemas.opeco_sabana_test_pred

    ) 

    select *
    from prediction