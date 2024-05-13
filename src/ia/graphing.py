import networkx as nx
import matplotlib.pyplot as plt


def generar_arbol(camino, titulo="Árbol de búsqueda"):
    plt.figure(figsize=(20, 20))
    G = nx.DiGraph()
    for estado, accion in camino.items():
        if accion:
            G.add_edge(accion[0].__repr__(), estado.__repr__(), label=accion[1])

    # pos = nx.spring_layout(G)
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    edge_labels = nx.get_edge_attributes(G, 'label')
    plt.title(titulo)
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="lightblue")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.savefig(f"{titulo}.png", dpi=300)
