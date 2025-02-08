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






drop table if exists {zona_p}.{prefijo}_sabana_trtest_hist_pagos purge;



create table if not exists {zona_p}.{prefijo}_sabana_trtest_hist_pagos stored as parquet tblproperties ('transactional'='false') as
    WITH trtest AS (
        -- Convierte la fecha a formato 'yyyy-mm-01' y resta 1 mes
        SELECT  
            nit_enmascarado,
            num_oblig_orig_enmascarado,
            num_oblig_enmascarado,
            var_rpta_alt,
            CAST(
                CONCAT(
                    SUBSTR(CAST(fecha_var_rpta_alt AS STRING), 1, 4), '-', 
                    SUBSTR(CAST(fecha_var_rpta_alt AS STRING), 5, 2), '-', 
                    '01'
                ) AS DATE
            ) - INTERVAL 1 MONTH AS fecha,
            CAST(
                CONCAT(
                    SUBSTR(CAST(fecha_var_rpta_alt AS STRING), 1, 4), '-', 
                    SUBSTR(CAST(fecha_var_rpta_alt AS STRING), 5, 2), '-', 
                    '01'
                ) AS DATE
            ) AS fecha_var_rpta_alt
        FROM proceso_ecosistemas.opeco_var_rpta_trtest
    ), 

    trtest_h AS (
        -- Genera el campo concatenado 'nit_oblig_fecha' y selecciona los campos necesarios
        SELECT 
            CONCAT(CAST(nit_enmascarado AS STRING), "#", CAST(num_oblig_enmascarado AS STRING),'#',CAST(fecha AS STRING)) AS nit_oblig_fecha,
            nit_enmascarado,
            num_oblig_orig_enmascarado,
            num_oblig_enmascarado,
            var_rpta_alt,
            fecha_var_rpta_alt,
            fecha
        FROM trtest
    ), 

    his_pagos AS (
        -- Convierte la fecha de corte a formato 'yyyy-mm-01' y la selecciona
        SELECT 
            *,
            CAST(
                CONCAT(
                    SUBSTR(CAST(fecha_corte AS STRING), 1, 4), '-', 
                    SUBSTR(CAST(fecha_corte AS STRING), 5, 2), '-', 
                    '01'
                ) AS DATE
            ) AS fecha
        FROM proceso_ecosistemas.opeco_maestro_cuotas_pagos
    ), 

    hist_pagos_h AS (
        -- Genera el campo concatenado 'nit_oblig_fecha' y selecciona los campos necesarios
        SELECT 
            CONCAT(CAST(nit_enmascarado AS STRING), "#", CAST(num_oblig_enmascarado AS STRING),'#',CAST(fecha AS STRING)) AS nit_oblig_fecha,
            nit_enmascarado,
            num_oblig_enmascarado,
            fecha,
            valor_cuota_mes,
            pago_total,
            marca_pago
        FROM his_pagos
    )

    -- Selección final con INNER JOIN para combinar los resultados
    SELECT 
        t1.nit_oblig_fecha AS key,
        t1.nit_enmascarado,
        t1.num_oblig_orig_enmascarado,
        t1.num_oblig_enmascarado,
        t1.var_rpta_alt,
        t1.fecha_var_rpta_alt,
        t2.fecha AS fecha_corte_lag,
        t2.valor_cuota_mes,
        t2.pago_total,
        t2.marca_pago
    FROM trtest_h AS t1
    INNER JOIN hist_pagos_h AS t2 
        ON t1.nit_oblig_fecha = t2.nit_oblig_fecha

--------------------------------- Query End ---------------------------------