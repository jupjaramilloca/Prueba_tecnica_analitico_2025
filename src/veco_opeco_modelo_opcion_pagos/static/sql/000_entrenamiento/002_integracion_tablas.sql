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
-- Escribe tu consulta SQL aquí ↓






drop table if exists {zona_p}.{prefijo}_entrenamiento_model_pagos

;

create table if not exists {zona_p}.{prefijo}_entrenamiento_model_pagos stored as parquet tblproperties ('transactional'='false') as


    with master as (

        select *,
            CAST(
            CONCAT(
                CAST(year AS STRING), '-', 
                CAST(month AS STRING), '-', 
                '01'
            ) AS DATE ) AS fecha_corte_ingestion,
            CAST(
                CONCAT(
                    SUBSTR(CAST(f_ult_mantenimiento AS STRING), 1, 4), '-', 
                    SUBSTR(CAST(f_ult_mantenimiento AS STRING), 5, 2), '-', 
                    '01'
                ) AS DATE
            ) AS fecha_corte,
        row_number() over (partition by nit_enmascarado order by year desc, month desc, ingestion_day desc) as r
        from proceso_ecosistemas.opeco_master_customer_data

    ), master_trtest_hist_pagos as (

        select 
            t1.*,
            t2.tipo_cli,
            t2.total_ing,
            t2.tot_activos,
            t2.segm,
            CASE 
                WHEN upper(t2.subsegm) IN ("PEQUENA","PEQUE#O") THEN "PEQUE#O"
                ELSE t2.subsegm
            END AS subsegm,
            t2.egresos_mes,
            t2.tot_patrimonio,
            t2.smmlv
        from proceso_ecosistemas.opeco_sabana_trtest_hist_pagos as t1
        inner join master as t2
        on t1.nit_enmascarado = t2.nit_enmascarado and 
        t2.r=1 --and 
        --t1.fecha_corte_lag = t2.fecha_corte

    ), prob as (

        select 
            *,
            CAST(
                CONCAT(
                    SUBSTR(CAST(fecha_corte AS STRING), 1, 4), '-', 
                    SUBSTR(CAST(fecha_corte AS STRING), 5, 2), '-', 
                    '01'
                ) AS DATE
            ) AS fecha
        from proceso_ecosistemas.opeco_probabilidad_oblig
        
    ), prob_h as (

        select 
            CONCAT(CAST(nit_enmascarado AS STRING), "#", CAST(num_oblig_enmascarado AS STRING),'#',CAST(fecha AS STRING)) AS nit_oblig_fecha,
            lote,
            prob_propension,
            prob_alrt_temprana,
            prob_auto_cura
        from prob

    ), master_trtest_hist_pagos_prob as (
    select t1.*,
    t2.lote,
    t2.prob_propension,
    t2.prob_alrt_temprana,
    t2.prob_auto_cura
    from master_trtest_hist_pagos as t1
    inner join prob_h as t2
    on t1.key = t2.nit_oblig_fecha

    )
    select *
    from master_trtest_hist_pagos_prob 