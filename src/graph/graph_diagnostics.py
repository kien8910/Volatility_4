from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch


def adjacency_to_edges(adjacency: torch.Tensor, tickers: list[str], model: str, fold_id: int, seed: int) -> pd.DataFrame:
    a = adjacency.detach().cpu().numpy()
    rows = []
    for i, src in enumerate(tickers):
        for j, dst in enumerate(tickers):
            if i != j and a[i, j] > 0:
                rows.append(
                    {
                        "model": model,
                        "fold_id": int(fold_id),
                        "seed": int(seed),
                        "source": src,
                        "target": dst,
                        "weight": float(a[i, j]),
                    }
                )
    return pd.DataFrame(rows)


def graph_stability(edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame(columns=["model", "source", "target", "selection_frequency", "mean_weight"])
    total = edges.groupby("model")[["fold_id", "seed"]].drop_duplicates().groupby("model").size()
    grouped = edges.groupby(["model", "source", "target"], as_index=False).agg(
        selections=("weight", "size"),
        mean_weight=("weight", "mean"),
    )
    grouped["selection_frequency"] = grouped.apply(lambda r: r["selections"] / total.loc[r["model"]], axis=1)
    return grouped.sort_values(["model", "selection_frequency", "mean_weight"], ascending=[True, False, False])


def plot_graph(adjacency: torch.Tensor, tickers: list[str], path: str | Path, title: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    a = adjacency.detach().cpu().numpy()
    graph = nx.DiGraph()
    for ticker in tickers:
        graph.add_node(ticker)
    for i, src in enumerate(tickers):
        for j, dst in enumerate(tickers):
            if i != j and a[i, j] > 0:
                graph.add_edge(src, dst, weight=float(a[i, j]))
    pos = nx.circular_layout(graph)
    weights = [max(0.5, 3.0 * graph[u][v]["weight"]) for u, v in graph.edges]
    plt.figure(figsize=(7, 7))
    nx.draw_networkx_nodes(graph, pos, node_color="#e8f1ff", edgecolors="#2d5aa7", node_size=900)
    nx.draw_networkx_labels(graph, pos, font_size=9)
    nx.draw_networkx_edges(graph, pos, width=weights, arrows=True, alpha=0.65)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_graph_stability(stability: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if stability.empty:
        plt.figure()
        plt.title("No graph stability data")
        plt.savefig(path)
        plt.close()
        return
    top = stability.head(30).copy()
    top["edge"] = top["source"] + "→" + top["target"]
    top.plot(kind="bar", x="edge", y="selection_frequency", figsize=(12, 5), legend=False)
    plt.title("Top graph edge selection frequencies")
    plt.xticks(rotation=70, ha="right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def mean_adjacency_from_edges(edges: pd.DataFrame, tickers: list[str], model: str) -> torch.Tensor | None:
    subset = edges.loc[edges["model"] == model]
    if subset.empty:
        return None
    pos = {ticker: idx for idx, ticker in enumerate(tickers)}
    adj = torch.zeros((len(tickers), len(tickers)), dtype=torch.float32)
    mean_edges = subset.groupby(["source", "target"], as_index=False)["weight"].mean()
    for row in mean_edges.itertuples(index=False):
        if row.source in pos and row.target in pos:
            adj[pos[row.source], pos[row.target]] = float(row.weight)
    return adj


def plot_unavailable_graph(tickers: list[str], path: str | Path, title: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    graph = nx.Graph()
    graph.add_nodes_from(tickers)
    pos = nx.circular_layout(graph)
    plt.figure(figsize=(7, 7))
    nx.draw_networkx_nodes(graph, pos, node_color="#f5f5f5", edgecolors="#999999", node_size=900)
    nx.draw_networkx_labels(graph, pos, font_size=9)
    plt.title(title)
    plt.text(0.5, -0.08, "No learned edge data available yet", ha="center", transform=plt.gca().transAxes)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
