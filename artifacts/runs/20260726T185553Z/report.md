# RAGForge benchmark report - 20260726T185553Z

## Retrieval

```
strategy        recall@k  precision@k   ndcg@k    mrr   drm@k    n errors
dense              0.972        0.270    0.957  0.974   0.014   57      0
sparse_bm25        0.918        0.239    0.857  0.868   0.074   57      0
hybrid_rrf         0.965        0.260    0.932  0.949   0.035   57      0
reranked           0.898        0.214    0.804  0.801   0.070   57      0
contextual         0.968        0.267    0.956  0.982   0.032   57      0
parent_child       0.972        0.301    0.957  0.974   0.015   57      0
sac                0.975        0.274    0.963  0.991   0.000   57      0
sac_contextual     0.967        0.277    0.953  0.978   0.000   57      0
raptor             1.000        0.611    0.945  0.988   0.004   57      0
graphrag           0.565        0.158    0.543  0.564   0.109   57      0
```

## Answer quality

```
strategy       citation_acc faithfulness  relevancy abstention    n errors
dense                 0.674        0.928      0.843      1.000   60      0
sparse_bm25           0.631        0.943      0.790      0.950   60      0
hybrid_rrf            0.674        0.951      0.834      0.967   60      0
reranked              0.667        0.903      0.786      0.883   60      0
contextual            0.678        0.968      0.840      0.983   60      0
parent_child          0.657        0.952      0.833      0.967   60      0
sac                   0.689        0.956      0.840      0.967   60      0
sac_contextual        0.676        0.952      0.808      0.983   60      0
raptor                0.657        0.953      0.826      1.000   60      0
graphrag              0.399        0.817      0.524      0.633   60      0
```
