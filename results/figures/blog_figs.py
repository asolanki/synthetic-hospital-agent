import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
SURF="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; GRID="#e4e3df"
BLUE="#2a78d6"; BLUED="#184f95"; GRAY="#b9b8b2"; ORANGE="#eb6834"; AQUA="#1baf7a"; YEL="#eda100"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":12,"axes.edgecolor":GRID,"text.color":INK,
 "axes.labelcolor":INK2,"xtick.color":INK2,"ytick.color":INK2,"figure.facecolor":SURF,"axes.facecolor":SURF})

# Figure 1: score ladder
rows=[("Qwen3-4B, untrained",0.17,"4b"),("+ SFT, 3 epochs",0.39,"4b"),("+ GRPO",0.40,"4b"),
("SFT, 8 epochs",0.53,"4b"),("Tool-SFT (taught to search)",0.74,"best4b"),
("GPT-6 Luna",0.69,"other"),("Kimi K2.5",0.73,"other"),("GPT-6 Luna + one paragraph",0.81,"luna")]
col={"4b":BLUE,"best4b":BLUED,"other":GRAY,"luna":ORANGE}
fig,ax=plt.subplots(figsize=(9,5.2),dpi=200)
y=list(range(len(rows)))[::-1]
for yi,(lab,v,k) in zip(y,rows):
    ax.barh(yi,v,height=0.62,color=col[k],edgecolor=SURF,linewidth=2)
    ax.text(max(v+0.012,0.832) if k=="luna" else v+0.012,yi,f"{v:.2f}",va="center",fontsize=11,color=INK,fontweight="bold" if k in("best4b","luna") else "normal")
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
ax.axhline(2.5,color=GRID,lw=1)
ax.axvline(0.82,color=INK2,lw=1.2,ls=(0,(4,3)))
ax.text(0.82,len(rows)-0.35,"copy-and-code ceiling 0.82",ha="right",va="bottom",fontsize=10,color=INK2)
ax.text(-0.005,7.45,"Qwen3-4B on one A100",fontsize=10,color=BLUED,ha="left",va="bottom",transform=ax.get_yaxis_transform())
ax.text(0.005,2.5,"Frontier models, no fine-tuning",fontsize=10,color=INK2,ha="left",va="center",backgroundcolor=SURF,transform=ax.get_yaxis_transform())
ax.set_xlim(0,0.9); ax.set_ylim(-0.6,len(rows)-0.1)
ax.xaxis.grid(True,color=GRID,lw=0.8); ax.set_axisbelow(True)
for s in ("top","right","left"): ax.spines[s].set_visible(False)
ax.tick_params(axis="y",length=0)
ax.set_xlabel("F1 on the 200 public test patients")
fig.suptitle("RL added 0.01. A search tool added 0.21. One paragraph took Luna to the top.",x=0.02,ha="left",fontsize=13.5,fontweight="bold")
fig.tight_layout(); fig.savefig("score-ladder.png",facecolor=SURF)

# Figure 2: accuracy by training exposure
x=["0\n(never seen)","1","2–4","5 or more"]
S={"SFT, 3 epochs":([5,28,37,73],BLUE,"o"),"+ GRPO":([5,24,39,73],YEL,"s"),
   "SFT, 8 epochs":([1,35,68,84],AQUA,"^"),"Tool-SFT":([57,83,87,96],ORANGE,"D")}
fig,ax=plt.subplots(figsize=(9,5.2),dpi=200)
for name,(v,c,m) in S.items():
    ax.plot(range(4),v,color=c,lw=2,marker=m,ms=8,markeredgecolor=SURF,markeredgewidth=2,label=name,zorder=3)
ax.annotate("1% → 57%",xy=(0,57),xytext=(0.18,57),fontsize=11,fontweight="bold",color=INK,va="center")
ax.annotate("",xy=(0,55),xytext=(0,3),arrowprops=dict(arrowstyle="->",color=INK2,lw=1.2))
ax.text(3.08,96,"Tool-SFT",va="center",fontsize=10.5,color=INK)
ax.text(3.08,84,"SFT, 8 epochs",va="center",fontsize=10.5,color=INK)
ax.text(3.08,73,"SFT 3 ep. / + GRPO",va="center",fontsize=10.5,color=INK)
ax.set_xticks(range(4)); ax.set_xticklabels(x); ax.set_xlim(-0.25,3.85); ax.set_ylim(0,102)
ax.set_yticks([0,25,50,75,100]); ax.set_yticklabels([f"{t}%" for t in [0,25,50,75,100]])
ax.yaxis.grid(True,color=GRID,lw=0.8); ax.set_axisbelow(True)
for s in ("top","right","left"): ax.spines[s].set_visible(False)
ax.tick_params(axis="y",length=0)
ax.set_xlabel("Training patients whose answer key carried the code")
ax.set_ylabel("Right ICD-10 category")
ax.legend(loc="lower right",frameon=False,fontsize=10.5,bbox_to_anchor=(0.86,0.02))
fig.suptitle("Training lifts codes the model has seen. The search tool lifts the ones it hasn't.",x=0.02,ha="left",fontsize=13.5,fontweight="bold")
fig.text(0.02,0.005,"Documented (earlier-visit) diagnoses, 200 public patients. RL's line sits on the 3-epoch fine-tune it started from.",fontsize=9,color=INK2)
fig.tight_layout(rect=(0,0.03,1,1)); fig.savefig("training-exposure.png",facecolor=SURF)

# Figure 3: RL learning signal by window (earlier-visit diagnoses, training patients)
W=["Updates 1–30","31–55","56–85"]
never=[33.1,33.8,34.5]; some=[37.5,33.8,25.0]; always=[29.4,32.4,40.4]
fig,ax=plt.subplots(figsize=(9,4.6),dpi=200)
import numpy as np
yy=np.arange(3)[::-1]
for vals,left,c,lab in [(never,[0]*3,GRAY,"Never right (0/4): no signal"),(some,never,BLUE,"Sometimes right: the only learnable case"),(always,[a+b for a,b in zip(never,some)],BLUED,"Always right (4/4): already solved")]:
    ax.barh(yy,vals,left=left,height=0.6,color=c,edgecolor=SURF,linewidth=2,label=lab)
    for y_,v,l in zip(yy,vals,left):
        ax.text(l+v/2,y_,f"{v:.0f}%",ha="center",va="center",fontsize=11,color=INK if c==GRAY else "white")
ax.set_yticks(yy); ax.set_yticklabels(W); ax.set_xlim(0,100); ax.set_xticks([])
for s_ in ("top","right","left","bottom"): ax.spines[s_].set_visible(False)
ax.tick_params(axis="y",length=0)
ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.02),ncol=3,frameon=False,fontsize=9.5)
fig.suptitle("GRPO turned \"sometimes\" into \"always\". \"Never\" never moved.",x=0.02,ha="left",fontsize=13.5,fontweight="bold")
fig.text(0.02,0.005,"Earlier-visit diagnoses on training patients, four rollouts each at temperature 1. Share of diagnoses by how many rollouts coded them right.",fontsize=9,color=INK2)
fig.tight_layout(rect=(0,0.04,1,1)); fig.savefig("rl-signal.png",facecolor=SURF)
