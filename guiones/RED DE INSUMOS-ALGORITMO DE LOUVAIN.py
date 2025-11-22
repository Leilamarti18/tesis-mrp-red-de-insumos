"""
Identificación de líderes por comunidad.

"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from community import community_louvain  # paquete python-louvain

# -------------------------------------------------------------------
# 1. Carga del CSV y limpieza de comas decimales
# -------------------------------------------------------------------

# Ruta del archivo de entrada
RUTA_CSV = "phi.csv"  # cámbialo si es necesario
UMBRAL = 0.5          # afinidad mínima para crear enlace

# Leer el CSV tratando de inferir el separador
df = pd.read_csv(RUTA_CSV, sep=None, engine="python")

# Asumimos que la primera columna contiene los nombres de los nodos
df = df.set_index(df.columns[0])

# Reemplazar comas decimales por puntos y convertir a numérico
df = df.applymap(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)
df = df.apply(pd.to_numeric, errors="coerce")

# -------------------------------------------------------------------
# 2. Verificar que el DataFrame sea una matriz cuadrada
#    (mismos índices y columnas) y recortarlo si hace falta
# -------------------------------------------------------------------

# Intersección de índices y columnas
comunes = df.index.intersection(df.columns)

# Recortar a la parte común (garantiza matriz cuadrada)
df = df.loc[comunes, comunes]

# Opcional: ordenar por nombre de nodo
df = df.sort_index()
df = df.loc[:, df.columns.sort_values()]

print(f"Matriz cuadrada final: {df.shape[0]} x {df.shape[1]}")

# -------------------------------------------------------------------
# 3. Construir la red no dirigida con enlaces >= 0.5
#    Recorriendo solo la parte superior de la matriz
# -------------------------------------------------------------------

G = nx.Graph()

# Añadir nodos (uno por cada índice)
nodos = list(df.index)
G.add_nodes_from(nodos)

# Crear enlaces solo por encima de la diagonal, con peso >= UMBRAL
for i, u in enumerate(nodos):
    for j in range(i + 1, len(nodos)):
        v = nodos[j]
        peso = df.loc[u, v]
        if pd.notna(peso) and peso >= UMBRAL:
            G.add_edge(u, v, weight=float(peso))

print(f"Número de nodos en la red: {G.number_of_nodes()}")
print(f"Número de enlaces en la red: {G.number_of_edges()}")

# -------------------------------------------------------------------
# 4. Detectar comunidades con Louvain y crear DataFrame de nodos
#    con: comunidad, grado, peso total y peso promedio
# -------------------------------------------------------------------

# Partición Louvain: diccionario nodo -> id_comunidad
particion = community_louvain.best_partition(G, weight="weight", random_state=42)
nx.set_node_attributes(G, particion, "comunidad")

datos_nodos = []

for nodo in G.nodes():
    comunidad = particion[nodo]
    grado = G.degree(nodo)  # grado no ponderado (número de vecinos)

    # Lista de pesos de las aristas incidentes al nodo
    pesos = [d.get("weight", 1.0) for _, _, d in G.edges(nodo, data=True)]
    peso_total = sum(pesos)
    peso_promedio = peso_total / grado if grado > 0 else 0.0

    datos_nodos.append(
        {
            "nodo": nodo,
            "comunidad": comunidad,
            "grado": grado,
            "peso_total": peso_total,
            "peso_promedio": peso_promedio,
        }
    )

df_nodos = pd.DataFrame(datos_nodos)

# -------------------------------------------------------------------
# 5. Identificar al líder de cada comunidad (nodo con mayor grado)
# -------------------------------------------------------------------

# Ordenamos por comunidad, luego por grado y peso_total (desc)
df_lideres = (
    df_nodos.sort_values(
        by=["comunidad", "grado", "peso_total"],
        ascending=[True, False, False],
    )
    .groupby("comunidad")
    .first()
    .reset_index()
)

df_lideres = df_lideres.rename(columns={"nodo": "lider"})
df_lideres["tamanio_comunidad"] = df_nodos.groupby("comunidad")["nodo"].transform("count").groupby(df_nodos["comunidad"]).first().values

# Marcar en df_nodos quién es líder
df_nodos = df_nodos.merge(df_lideres[["comunidad", "lider"]], on="comunidad", how="left")
df_nodos["es_lider"] = df_nodos["nodo"] == df_nodos["lider"]

# -------------------------------------------------------------------
# 6. Metagrafo entre comunidades basado en enlaces cruzados
# -------------------------------------------------------------------

# Nodos del metagrafo = ids de comunidades
comunidades = sorted(df_nodos["comunidad"].unique())
meta_G = nx.Graph()
meta_G.add_nodes_from(comunidades)

# Construimos aristas entre comunidades si hay enlaces cruzados
for u, v, d in G.edges(data=True):
    cu = particion[u]
    cv = particion[v]
    if cu == cv:
        continue  # enlaces internos de comunidad no van al metagrafo

    peso = d.get("weight", 1.0)

    if meta_G.has_edge(cu, cv):
        meta_G[cu][cv]["weight"] += peso
        meta_G[cu][cv]["count"] += 1
    else:
        meta_G.add_edge(cu, cv, weight=peso, count=1)

print(f"Número de comunidades: {meta_G.number_of_nodes()}")
print(f"Número de enlaces entre comunidades: {meta_G.number_of_edges()}")

# -------------------------------------------------------------------
# 7. Detectar “familias” de comunidades aplicando Louvain al metagrafo
# -------------------------------------------------------------------

# Si el metagrafo tiene al menos una arista, aplicamos Louvain.
# Si no, cada comunidad será su propia familia.
if meta_G.number_of_edges() > 0:
    familias = community_louvain.best_partition(meta_G, weight="weight", random_state=42)
else:
    familias = {c: c for c in comunidades}

# familias: comunidad -> familia
# Añadimos la familia al info de nodos
df_nodos["familia"] = df_nodos["comunidad"].map(familias)

# DataFrame con comunidades y sus miembros
df_comunidades = (
    df_nodos[["comunidad", "familia", "nodo"]]
    .sort_values(["familia", "comunidad", "nodo"])
    .reset_index(drop=True)
)

# DataFrame con familias y comunidades (lista única)
df_familias_com = (
    df_comunidades[["familia", "comunidad"]]
    .drop_duplicates()
    .sort_values(["familia", "comunidad"])
    .reset_index(drop=True)
)

# -------------------------------------------------------------------
# 8. Exportar resultados a un archivo Excel con múltiples hojas
# -------------------------------------------------------------------

RUTA_EXCEL = "analisis_red_phi.xlsx"

with pd.ExcelWriter(RUTA_EXCEL, engine="openpyxl") as writer:
    # Detalle de nodos con características
    (
        df_nodos.sort_values(["familia", "comunidad", "nodo"])
        .to_excel(writer, sheet_name="Detalle_nodos", index=False)
    )

    # Listado de comunidades con sus miembros
    df_comunidades.to_excel(writer, sheet_name="Comunidades", index=False)

    # Líderes por comunidad
    (
        df_lideres.sort_values("comunidad")
        .to_excel(writer, sheet_name="Lideres_comunidad", index=False)
    )

    # Familias detectadas entre comunidades
    df_familias_com.to_excel(writer, sheet_name="Familias_comunidades", index=False)

print(f"Archivo Excel generado: {RUTA_EXCEL}")

# -------------------------------------------------------------------
# 9. Visualización mejorada de la red original
#    - Filtra componentes pequeñas (ruido)
#    - Layout por comunidades (clusters separados)
#    - Colores Tableau por comunidad
#    - Tamaño de nodos proporcional al grado
#    - Líderes resaltados y etiquetados
#    - Grosor de aristas según peso
#    - Layout alternativo kamada_kawai (figura 2)
# -------------------------------------------------------------------

import numpy as np
import matplotlib.colors as mcolors

# --- 9.0. Preparar información básica ---

# Partición nodo -> comunidad (ya la tienes en 'particion'; por si acaso la leemos de G)
if not 'particion' in locals():
    particion = nx.get_node_attributes(G, "comunidad")

# Lista de líderes (ya calculada en df_nodos)
lideres = df_nodos[df_nodos["es_lider"]]["nodo"].tolist()

# --- 9.1. Filtrar componentes pequeñas (dejar solo las que tienen >= 3 nodos) ---

componentes = [G.subgraph(c).copy() for c in nx.connected_components(G)]
componentes_grandes = [c for c in componentes if len(c.nodes()) >= 3]

if componentes_grandes:
    G_plot = nx.compose_all(componentes_grandes)
else:
    # Si todas son pequeñas, usamos la red completa
    G_plot = G.copy()

print(f"Nodos mostrados en la figura: {G_plot.number_of_nodes()}")

# --- 9.2. Layout por comunidades, separadas horizontalmente ---

comunidades_ordenadas = sorted({particion[n] for n in G_plot.nodes()})
pos = {}
offset_x = 0.0
spacing_x = 4.0  # separación horizontal entre comunidades

for com in comunidades_ordenadas:
    nodos_com = [n for n in G_plot.nodes() if particion[n] == com]
    if not nodos_com:
        continue

    subG = G_plot.subgraph(nodos_com)
    # layout interno de la comunidad (spring layout mejorado)
    pos_sub = nx.spring_layout(subG, seed=42, k=0.8, iterations=200)

    # desplazar todo el cluster hacia la derecha para separarlo del anterior
    for n, (x, y) in pos_sub.items():
        pos[n] = np.array([x + offset_x, y])

    offset_x += spacing_x

# Si por alguna razón pos quedó vacío, usamos un spring_layout global
if not pos:
    pos = nx.spring_layout(G_plot, seed=42, k=1.2, iterations=200)

# --- 9.3. Colores por comunidad (paleta Tableau, alto contraste) ---

colors = list(mcolors.TABLEAU_COLORS.values())
color_por_comunidad = {
    com: colors[i % len(colors)]
    for i, com in enumerate(comunidades_ordenadas)
}

node_colors = [color_por_comunidad[particion[n]] for n in G_plot.nodes()]

# --- 9.4. Tamaño de nodos proporcional al grado ---

deg = dict(G_plot.degree())
max_deg = max(deg.values()) if deg else 1
min_size = 200
max_size = 1200

node_sizes = [
    min_size + (deg[n] / max_deg) * (max_size - min_size)
    for n in G_plot.nodes()
]

# Tamaño extra para líderes
lideres_plot = [n for n in lideres if n in G_plot.nodes()]
sizes_lideres = [
    node_sizes[list(G_plot.nodes()).index(n)] * 1.4
    for n in lideres_plot
]

# --- 9.5. Grosor de aristas según el peso ---

weights = [d.get("weight", 1.0) * 2 for _, _, d in G_plot.edges(data=True)]

# --- 9.6. Figura principal: layout por comunidades ---

plt.figure(figsize=(14, 10))

# Aristas
nx.draw_networkx_edges(
    G_plot,
    pos,
    width=weights,
    alpha=0.35
)

# Nodos
nx.draw_networkx_nodes(
    G_plot,
    pos,
    node_color=node_colors,
    node_size=node_sizes,
    alpha=0.9,
)

# Líderes resaltados (borde negro y un poco más grandes)
if lideres_plot:
    nx.draw_networkx_nodes(
        G_plot,
        pos,
        nodelist=lideres_plot,
        node_size=sizes_lideres,
        node_color=[color_por_comunidad[particion[n]] for n in lideres_plot],
        edgecolors="black",
        linewidths=2,
        alpha=1.0,
    )

# Etiquetas SOLO para líderes (para no saturar)
labels = {n: n for n in lideres_plot}
nx.draw_networkx_labels(
    G_plot,
    pos,
    labels=labels,
    font_size=7,
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7)
)

plt.title("Red de afinidad entre insumos por comunidades (Louvain)")
plt.axis("off")
plt.tight_layout()
plt.show()

# --- 9.7. Figura alternativa: layout kamada_kawai (misma red filtrada) ---

plt.figure(figsize=(14, 10))

pos2 = nx.kamada_kawai_layout(G_plot)

nx.draw_networkx_edges(
    G_plot,
    pos2,
    width=weights,
    alpha=0.35
)

nx.draw_networkx_nodes(
    G_plot,
    pos2,
    node_color=node_colors,
    node_size=node_sizes,
    alpha=0.9,
)

if lideres_plot:
    nx.draw_networkx_nodes(
        G_plot,
        pos2,
        nodelist=lideres_plot,
        node_size=sizes_lideres,
        node_color=[color_por_comunidad[particion[n]] for n in lideres_plot],
        edgecolors="black",
        linewidths=2,
        alpha=1.0,
    )

labels2 = {n: n for n in lideres_plot}
nx.draw_networkx_labels(
    G_plot,
    pos2,
    labels=labels2,
    font_size=7,
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7)
)

plt.title("Red de afinidad entre insumos (layout Kamada-Kawai)")
plt.axis("off")
plt.tight_layout()
plt.show()
