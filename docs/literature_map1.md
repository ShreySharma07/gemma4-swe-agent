# Map 1 — Literature grid (27 Sep 2026)

Goal: check which of our four levers are already published.
Levers:
1. issue text → entry-node grounding
2. restoring direction + typed edges
3. structure-aware ranking / GNN
4. training the model to navigate the graph

| Paper | Graph | Entry point | Trained? | Models | Takeaway |
|---|---|---|---|---|---|
| **LocAgent** (2503.09089, ACL'25) | dir/file/class/function; contain, import, invoke, inherit; directed + typed | hierarchical index: exact ID → name dict → BM25 over IDs → BM25 over code | SFT (LoRA) on 768 successful trajectories, distilled to 7B | Qwen2.5-Coder 7B/32B | 32B-ft ≈ Claude-3.5 on localisation. Accuracy falls with hop distance. Tree-formatted subgraph output beats DOT/JSON. |
| **RepoGraph** (2410.14684, ICLR'25) | line-level def/ref; invoke + contain | agent-chosen term → k-hop ego graph | No | GPT-4/4o, Claude-3.5 | +32.8% relative as a plug-in. 2-hop is worse than baseline (noise); 1-hop is best. |
| **CodexGraph** (2408.03910) | module/class/function; contain + invoke | LLM writes Cypher | No | GPT-4o | Depends on the LLM's query skill. |
| **LARGER** (2605.16352) | dir/file/class/func; contains, imports, invokes, code–test; directed, weighted | lexical anchoring: grep hits → graph nodes → neighbours injected into the same observation | No | GPT-5.2 | +11.8–13.9 Acc@5. Argues separate graph tools fragment the agent loop. |
| **ARISE** (2605.03117) | Contains, Imports, Calls/CalledBy, Inherits, def-use | TF-IDF + suspect-region ranking | No | Qwen2.5-Coder-32B | +4.7 pp resolve. Data-flow slicing gives the largest gain. Spurious edges hurt more than missing ones. |
| **ToolTrain / RepoSearcher** (2508.03012) | none (name lookup tools) | agent reasons | rejection-sampled SFT → RL (nDCG@5) on 28k issue–PR examples | Qwen 7B/32B | ≈ Claude-3.7 at function level. SFT → RL is best. |
| **CodeScout** (2603.17829) | none, bash only | agent greps | pure RL (GSPO), file/module/function F1 reward, 39k SWE-smith | Qwen3 1.7–14B | Beats graph-tool agents with 2–18× smaller models. Challenges whether graph tools are needed after training. |
| **GREPO** (2602.13921) | dir/file/class/func; contain, call, inherit + temporal | query rewrite → dense anchors → k-hop | GNNs on 47k tasks | GAT etc. | GNNs beat LLM agents on func/class Hit@k. |

Still to read: One Tool Is Enough (2512.20957), A²Agent (2608.29831), SweRank (2505.07849), SWE-Explore (2606.07297), RepoNav (2609.08355), CGM (2505.16901), CoSIL, OrcaLoca, Agentless.

## Verdict per lever
1. **Entry grounding:** largely taken (LocAgent, LARGER, GREPO). Useful for the leaderboard, weak as headline novelty.
2. **Direction + typed edges:** an open gap as far as found, but judged too obvious as a headline.
3. **GNN / structure ranking:** partly taken (GREPO).
4. **Training to navigate:** crowded (LocAgent, ToolTrain, CodeScout). Open question: does post-training make graph tools unnecessary under a fixed, imperfect graph and a quantized 31B model?
