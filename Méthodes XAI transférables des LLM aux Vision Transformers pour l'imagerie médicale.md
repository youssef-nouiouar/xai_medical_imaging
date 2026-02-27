# Méthodes XAI transférables des LLM aux Vision Transformers pour l'imagerie médicale

Les méthodes d'explicabilité développées pour les Large Language Models offrent un potentiel de transfert significatif vers les Vision Transformers utilisés en imagerie médicale. **AttnLRP** (ICML 2024) et la méthode **Chefer et al.** (CVPR 2021) représentent les approches les plus matures et validées pour ce transfert, tandis que les **Sparse Autoencoders** constituent la frontière la plus prometteuse de la recherche en interprétabilité mécanistique pour ViT. Ce rapport synthétise l'état de l'art 2023-2025, identifie les adaptations nécessaires pour chaque catégorie de méthodes, et fournit des recommandations concrètes pour une thèse sur l'IA explicable en imagerie médicale.

---

## Méthodes basées sur l'attention : du texte vers les patchs d'images

L'attention multi-têtes constitue le mécanisme central partagé entre LLM et ViT, rendant ces méthodes particulièrement adaptées au transfert. La **visualisation brute de l'attention** (Vaswani et al., NeurIPS 2017) projette directement les poids d'attention sous forme de heatmaps, mais souffre de limitations majeures : elle est **class-agnostic** et ne reflète qu'une seule couche. Jain & Wallace (NAACL 2019) ont démontré que les poids d'attention sont souvent décorrélés de l'importance réelle des tokens.

**Attention Rollout** (Abnar & Zuidema, ACL 2020) résout partiellement ce problème en propageant l'attention à travers les couches via multiplication matricielle : `AttentionRollout_L = Ã_L × AttentionRollout_{L-1}` où `Ã = 0.5 × A + 0.5 × I` intègre les connexions résiduelles. Pour les ViT, l'implémentation de **jacobgil/vit-explain** recommande d'utiliser la fusion par maximum plutôt que par moyenne, et un ratio de discard de **90%** pour réduire le bruit.

La méthode **Chefer et al.** (CVPR 2021) représente l'avancée la plus significative : elle combine LRP, gradients pondérés et rollout en trois étapes : (1) calcul de pertinence locale par Deep Taylor Decomposition, (2) moyennage des têtes d'attention pondéré par les gradients `Ā = E_h[∇A ⊙ A]⁺`, (3) agrégation par rollout avec pertinence. Cette méthode est **class-specific** et a été validée sur des datasets médicaux (COVID-19 chest X-rays, polyps coliques, tumeurs mammaires).

**AttCAT** (Qiang et al., NeurIPS 2022) et **LeGrad** (2024) représentent les évolutions récentes. LeGrad utilise directement le gradient par rapport aux cartes d'attention comme signal d'explicabilité, offrant une meilleure fidélité spatiale pour la segmentation. Le dépôt **WalBouss/LeGrad** fournit une implémentation compatible avec SigLIP et les poolers attentionnels.

---

## Propagation de pertinence : AttnLRP et règles spécifiques aux Transformers

La Layer-wise Relevance Propagation (LRP), initialement développée par Bach et al. (PLOS ONE 2015), a nécessité des adaptations substantielles pour les Transformers. Le défi principal réside dans la propagation à travers le **softmax** et la **LayerNorm**.

**AttnLRP** (Achtibat et al., ICML 2024) constitue l'état de l'art actuel. La méthode introduit une décomposition du softmax via une proposition mathématique élégante : `Rᵢˡ = xᵢ(Rᵢˡ⁺¹ - sᵢΣⱼRⱼˡ⁺¹)` où un terme de biais caché absorbe la pertinence, assurant la stabilité numérique. Pour la multiplication matricielle de l'attention, la pertinence est distribuée également entre les branches attention (A) et valeurs (V). La complexité est **O(1) en temps** et **O(√N) en mémoire**.

| Méthode LRP | Année | Venue | Modèles validés | Support ViT |
|-------------|-------|-------|-----------------|-------------|
| AttnLRP | 2024 | ICML | LLaMA 2, Flan-T5, Mixtral, ViT | Natif |
| Conservative Propagation | 2022 | ICML | BERT, DistilBERT, Graphormer | Adaptable |
| PA-LRP | 2025 | NeurIPS | LLaMA 2/3, DeiT | Natif |

**PA-LRP** (Bakish et al., NeurIPS 2025) apporte une innovation majeure : la propagation de pertinence à travers les **encodages positionnels**. L'étude révèle que l'information positionnelle contribue à **20-35%** des décisions du modèle, particulièrement pertinent pour l'imagerie médicale où la localisation des lésions est critique.

Pour l'adaptation aux ViT médicaux, la bibliothèque **LXT** (github.com/rachtibat/LRP-eXplains-Transformers) recommande d'utiliser la **γ-rule avec γ≈0.25** pour réduire le bruit caractéristique des Vision Transformers ("gradient shattering").

---

## Analyse des représentations : probing classifiers et CKA pour comprendre les couches

Les méthodes d'analyse des représentations permettent de comprendre **quelles informations** sont encodées à chaque couche du réseau, une question fondamentale pour l'interprétabilité en imagerie médicale.

Les **Probing Classifiers** (Belinkov, Computational Linguistics 2022) entraînent un classifieur linéaire sur les représentations gelées pour prédire des propriétés spécifiques. La métrique de **sélectivité** (Hewitt & Liang, EMNLP 2019) distingue l'information véritablement encodée de la mémorisation : `SEL = PERF(probe) - PERF(probe_random)`. Pour l'imagerie médicale, les concepts à sonder incluent :

- **Structures anatomiques** : frontières d'organes, types tissulaires
- **Caractéristiques pathologiques** : présence/type de tumeur, calcifications
- **Patterns de texture** : homogénéité, granularité, netteté des bords
- **Relations spatiales** : positions relatives, symétrie

**Centered Kernel Alignment (CKA)** (Kornblith et al., ICML 2019) mesure la similarité entre représentations via le critère HSIC : `CKA(K,L) = HSIC(K,L) / √(HSIC(K,K) × HSIC(L,L))`. Cette méthode est invariante aux transformations orthogonales et au scaling isotrope. **Raghu et al.** (NeurIPS 2021) ont démontré que les ViT obtiennent des représentations globales dès les couches superficielles, contrairement aux CNN.

Les **Concept Bottleneck Models** (Koh et al., ICML 2020) structurent explicitement le réseau pour prédire d'abord des **concepts humains compréhensibles**, puis la classe finale. Cette approche a été validée sur le **grading d'arthrose** (concepts : ostéophytes, rétrécissement articulaire) et permet l'**intervention à l'inférence** (correction des prédictions de concepts par un expert).

---

## Attribution de features : Integrated Gradients et SHAP pour Transformers

Les méthodes d'attribution quantifient la contribution de chaque entrée (patch) à la prédiction finale, fondamentales pour la validation clinique.

**Integrated Gradients** (Sundararajan et al., ICML 2017) satisfait des axiomes théoriques forts (sensibilité, invariance d'implémentation, complétude) : `IG_i(x) = (x_i - x'_i) × ∫₀¹ (∂F(x' + α(x - x'))) / (∂x_i) dα`. Le choix de la **baseline** est critique pour l'imagerie médicale : une image noire peut manquer les features sombres. Les alternatives recommandées sont le **flou gaussien**, la **moyenne du domaine** (tissu sain typique), ou **Expected Gradients** (moyenne sur une distribution de baselines).

**ViT-Shapley** (Covert et al., ICML 2022) résout le problème de complexité exponentielle des valeurs de Shapley par un **explainer amortisé** : un modèle secondaire apprend à prédire les attributions en un seul passage forward. L'implémentation (github.com/suinleelab/vit-shapley) utilise le masquage d'attention pour évaluer le ViT avec information partielle.

| Méthode | Complexité | Fidélité | Usage médical |
|---------|------------|----------|---------------|
| Integrated Gradients | O(steps × forward) | Haute | Classification, baseline critique |
| ViT-Shapley | O(1) après entraînement | Très haute | Production, temps réel |
| GradCAM (Chefer) | O(1) | Moyenne-haute | Visualisation rapide |
| Occlusion | O(n_patches × forward) | Haute | Validation, lent |

**pytorch-grad-cam** (github.com/jacobgil/pytorch-grad-cam) fournit une implémentation prête à l'emploi avec des **reshape transforms** pour ViT :
```python
def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, -1)
    return result.permute(0, 3, 1, 2)
```

---

## Interprétabilité mécanistique : circuits et Sparse Autoencoders

L'interprétabilité mécanistique vise à comprendre **comment** les modèles calculent leurs prédictions, au-delà de **quelles** features sont importantes.

L'**analyse de circuits** (Elhage, Olah, Anthropic 2021-2025) décompose les Transformers en graphes computationnels interprétables. Les circuits **QK** (Query-Key) déterminent quels tokens s'attendent mutuellement, tandis que les circuits **OV** (Output-Value) décrivent quelle information est copiée. Pour les LLM, des circuits comme les **induction heads** (match-and-copy) ont été identifiés comme mécanisme principal du few-shot learning.

**Activation Patching** (Meng et al., NeurIPS 2022 - ROME) identifie les activations causalement nécessaires :
1. Exécution avec entrée propre → enregistrement des activations
2. Exécution avec entrée corrompue (bruit sur certaines régions)
3. **Patch** des activations propres vers la version corrompue
4. Identification des patchs qui restaurent le comportement

Pour l'imagerie médicale, cette approche permettrait d'identifier quels **patchs anatomiques** sont causalement importants pour la segmentation tumorale dans UNETR.

Les **Sparse Autoencoders (SAEs)** (Anthropic 2023-2024) représentent l'avancée majeure récente. Ils résolvent le problème de **polysémantie** (neurones répondant à plusieurs concepts non-liés) en encodant vers une représentation sparse sur-complète. Les features extraites sont **monosémantiques** et interprétables. **ViT-Prisma** (Sonia Joseph, 2024) étend cette approche aux ViT :

- **PatchSAE** : attributions spatiales au niveau patch pour CLIP ViT
- **Matryoshka SAE** : représentations hiérarchiques multi-granularité
- **80+ poids SAE pré-entraînés** disponibles pour CLIP, DINO, V-JEPA

**TCAV** (Kim et al., ICML 2018) offre une approche plus accessible : les **Concept Activation Vectors** représentent des directions dans l'espace d'activation correspondant à des concepts humains. Le score TCAV quantifie l'importance du concept pour une classe. Cette méthode a été **validée en imagerie médicale** pour la rétinopathie diabétique et permet de détecter les **corrélations spurieuses** (artefacts de scanner).

---

## Outils et bibliothèques Python par catégorie

| Bibliothèque | URL | Méthodes supportées | Support ViT médical |
|--------------|-----|---------------------|---------------------|
| **Captum** | captum.ai | IG, SHAP, Occlusion, LIME, LRP basique | ✓ Classification |
| **LXT** | github.com/rachtibat/LRP-eXplains-Transformers | AttnLRP, pertinence neuronale | ✓ ViT, adaptable UNETR |
| **pytorch-grad-cam** | github.com/jacobgil/pytorch-grad-cam | GradCAM++, ScoreCAM, EigenCAM | ✓ ViT, Swin, DeiT |
| **Transformer-Explainability** | github.com/hila-chefer/Transformer-Explainability | Chefer LRP+attention | ✓ ViT, DeiT, BERT |
| **ViT-Prisma** | github.com/soniajoseph/ViT-Prisma | SAE, activation patching | ✓ CLIP, DINO |
| **pyvene** | github.com/stanfordnlp/pyvene | Interventions causales | Adaptable |
| **TransformerLens** | transformerlensorg.github.io | Circuits, logit lens | LLM seulement |
| **Zennit** | github.com/chr5tphr/zennit | LRP composites | ✓ Configurable |

---

## Tableau comparatif des méthodes XAI pour Vision Transformers médicaux

| Méthode | Principe | Adaptabilité ViT | Modifications requises | Références/Code |
|---------|----------|------------------|------------------------|-----------------|
| **Attention Rollout** | Multiplication récursive des matrices d'attention à travers les couches avec identité pour les résidus | ✓✓✓ Directe | Reshape 1D→2D (14×14 pour 224×224), fusion heads par max, discard 90% | Abnar & Zuidema, ACL 2020 • jacobgil/vit-explain |
| **Chefer et al.** | LRP + gradients pondérés + rollout : `Ā = E_h[∇A ⊙ A]⁺` agrégé par couche | ✓✓✓ Natif ViT/DeiT | Aucune pour ViT standard ; extension nécessaire pour Swin (fenêtres) | Chefer et al., CVPR 2021 • hila-chefer/Transformer-Explainability |
| **AttnLRP** | Décomposition Taylor du softmax, règles spécifiques pour LayerNorm et attention | ✓✓✓ Natif | γ-rule (γ≈0.25) pour ViT ; extension 3D pour UNETR | Achtibat et al., ICML 2024 • rachtibat/LRP-eXplains-Transformers |
| **PA-LRP** | Propagation de pertinence à travers encodages positionnels (learned, RoPE) | ✓✓ DeiT validé | Adapter règles PE selon architecture (sinusoïdal vs appris vs RoPE) | Bakish et al., NeurIPS 2025 • YardenBakish/PE-AWARE-LRP |
| **Integrated Gradients** | Intégrale du gradient sur chemin baseline→input : axiomes complétude/sensibilité | ✓✓✓ Via Captum | Baseline adaptée (moyenne tissu sain, flou) ; attribution couche patch_embed | Sundararajan et al., ICML 2017 • pytorch/captum |
| **ViT-Shapley** | Valeurs Shapley amorties via explainer entraîné sur masquage d'attention | ✓✓✓ Conçu pour ViT | Entraînement explainer sur dataset médical | Covert et al., ICML 2022 • suinleelab/vit-shapley |
| **GradCAM-Transformer** | Gradients w.r.t. activations pondérés et projetés spatialement | ✓✓ Avec reshape | Target layer = blocks[-2].norm1 ; reshape transform CLS→patches | Selvaraju et al., ICCV 2017 adapté • jacobgil/pytorch-grad-cam |
| **Probing Classifiers** | Classifieur linéaire sur représentations gelées pour prédire propriétés | ✓✓✓ Agnostique | Définir concepts médicaux (structures anatomiques, pathologies) | Belinkov, Comp. Ling. 2022 • john-hewitt/structural-probes |
| **CKA** | Similarité de représentations via HSIC normalisé : invariant scale/rotation | ✓✓✓ Agnostique | Aucune ; comparer couches, architectures, pré-entraînements | Kornblith et al., ICML 2019 • yuanli2333/CKA-Centered-Kernel-Alignment |
| **Concept Bottleneck Models** | Prédiction intermédiaire de concepts humains avant classe finale | ✓✓ Architecture | Remplacer head par couche concepts→labels ; annotations concepts requises | Koh et al., ICML 2020 • mertyg/post-hoc-cbm |
| **TCAV** | Direction CAV dans espace activations représentant concept ; score TCAV = importance | ✓✓✓ Validé médical | Créer datasets concepts (lésions, textures, organes) | Kim et al., ICML 2018 • tensorflow/tcav |
| **Sparse Autoencoders** | Encodage sparse sur-complet pour extraire features monosémantiques | ✓✓ Recherche active | Entraîner SAE sur activations ViT médical ; visualisation features | Anthropic 2023-24, Joseph 2024 • soniajoseph/ViT-Prisma |
| **Activation Patching** | Intervention causale : patch activations clean→corrupted pour identifier composants suffisants | ✓✓ Adaptable | Définir corruption régionale (pas token) ; 3D pour volumes | Meng et al., NeurIPS 2022 • stanfordnlp/pyvene |
| **Logit Lens** | Projection couches intermédiaires vers espace sortie pour suivre évolution prédiction | ✓✓ Appliqué ViT | Utiliser espace classes au lieu de vocabulaire ; segmentation = par patch | nostalgebraist 2020, Belrose 2023 • tuned-lens |

---

## Applications spécifiques à l'imagerie médicale

### Dermatologie (ISIC dataset)
**SkinSwinViT** (MDPI 2024) atteint **97.88%** de précision sur ISIC 2018 avec un mécanisme d'attention cross-window. L'étude de Barekatain & Glocker (2025) démontre que **DINO + Grad-CAM** fournit les explications les plus fidèles et localisées pour la classification de lésions cutanées. Pour les benchmarks ISIC, les méthodes recommandées sont Chefer et al. pour les visualisations class-specific et TCAV pour valider l'utilisation des critères ABCDE (Asymétrie, Bord, Couleur, Diamètre, Évolution).

### Radiologie thoracique
**xViTCOS** (IEEE JBHI 2021) propose un ViT explicable pour le screening COVID-19 via CXR. L'évaluation de Komorowski et al. (CVPR Workshop 2023) sur chest X-rays montre que **LRP pour Transformers surpasse LIME et la visualisation d'attention** en fidélité, sensibilité et complexité. L'approche hybride **ConvNeXtV2-ViT avec raisonnement neuro-symbolique** (Diagnostics 2026) intègre des ontologies cliniques aux explications Grad-CAM.

### Segmentation tumorale (UNETR/Swin UNETR)
**UNETR** (Hatamizadeh et al., WACV 2022) combine un encodeur Transformer avec un décodeur CNN pour la segmentation 3D. Les skip connections des couches 3, 6, 9, 12 nécessitent une **attribution bi-directionnelle**. Pour l'explicabilité :
- **Encodeur** : Layer-wise Integrated Gradients sur les sorties z3, z6, z9, z12
- **Décodeur** : GradCAM sur les feature maps convolutionnelles
- **Attention 3D** : Rollout avec reshape volumétrique (D', H', W')

**UNETR++** (IEEE TMI 2024) introduit l'Efficient Paired Attention avec complexité linéaire, atteignant **87.2% Dice** sur Synapse. L'interprétabilité des mécanismes d'attention efficaces reste un problème ouvert.

---

## Benchmarks et métriques d'évaluation XAI

**M⁴ Benchmark** (NeurIPS 2023) unifie l'évaluation XAI pour images et texte avec une taxonomie de métriques de fidélité :
1. **Sans ground truth** : courbes insertion/deletion, ABPC
2. **Ground truth synthétique** : attributions générées
3. **Annotations humaines** : alignement avec experts

Pour l'imagerie médicale, les métriques spécifiques incluent :
- **Localisation** : IoU avec régions annotées par radiologues
- **Plausibilité** : pertinence clinique des régions saillantes
- **Confiance-corrélation** : performance de localisation vs confiance du modèle

L'étude de Barekatain & Glocker (2025) révèle que **toutes les méthodes de saillance performent significativement moins bien que les experts humains** pour la localisation de pathologies, particulièrement pour les petites lésions et les formes complexes.

---

## Publications récentes et directions de recherche (2023-2025)

### Surveys clés
- **"A Practical Review of Mechanistic Interpretability"** (Rai et al., ICML 2025) : taxonomie task-centric, feuille de route pour débutants
- **"Advances in Medical Image Analysis with Vision Transformers"** (Azad et al., Medical Image Analysis 2024) : 200+ papiers sur ViT médicaux
- **"Bridging the Black Box"** (Somvanshi et al., 2025) : organisation en trois niveaux (neurones, circuits, algorithmes)

### Problèmes ouverts identifiés
1. **Scalabilité de l'interprétabilité mécanistique** : les circuits identifiés sur petits modèles généralisent-ils ?
2. **Transfert cross-modal** : attention causale (LLM) vs bidirectionnelle (ViT)
3. **Validation clinique** : gap entre performance technique XAI et utilité/confiance clinique
4. **Superposition** : représentation de plus de features que de neurones disponibles

### Workshops pertinents (2024-2025)
- **ICML Mechanistic Interpretability Workshop 2024**
- **eXCV Workshop: Explainable AI for Computer Vision** (ICCV 2025)
- **4th Workshop on Transformers for Vision** (CVPR 2025)
- **Skin Image Analysis Workshop** (MICCAI 2024) avec challenges ISIC

---

## Recommandations pour une thèse en XAI pour ViT médicaux

### Protocole d'évaluation recommandé
1. **Baseline** : Attention Rollout (rapide, établi)
2. **Méthode principale** : Chefer et al. ou AttnLRP (class-specific, validé médical)
3. **Attribution axiomatique** : Integrated Gradients avec baseline domaine-spécifique
4. **Validation concepts** : TCAV avec concepts cliniques (ABCDE pour dermato, consolidation/épanchement pour thorax)
5. **Analyse représentations** : CKA pour comparer pré-entraînements (ImageNet vs RadImageNet)

### Architecture → Méthode optimale

| Architecture | Méthode recommandée | Couche cible | Handling spécial |
|--------------|---------------------|--------------|------------------|
| ViT-B/16 | Chefer / AttnLRP | blocks[-2] | CLS attention extraction |
| Swin | GradCAM adapté | layers[-1].blocks[-1] | Agrégation attention fenêtres |
| DeiT | Gradient Rollout | Similar ViT | Token distillation à gérer |
| UNETR | LayerIG + GradCAM | encoder.layers[3,6,9,12] | 3D reshape, skip connections |

### Contribution potentielle
L'absence de travaux d'interprétabilité mécanistique spécifiques à l'imagerie médicale représente une **opportunité de recherche significative**. Une contribution pourrait porter sur :
- Entraînement de **SAE médicaux** sur UNETR/Swin UNETR pour découvrir des features anatomiques interprétables
- **Activation patching volumétrique** pour identifier les régions 3D causalement importantes
- **Logit lens médical** suivant l'évolution de la confiance diagnostique à travers les couches
- Benchmark unifié XAI pour ViT médicaux avec annotations radiologues

---

## Conclusion

Le transfert des méthodes XAI des LLM vers les Vision Transformers médicaux est **techniquement mature** pour les approches basées sur l'attention (Chefer, AttnLRP) et l'attribution de features (Integrated Gradients, TCAV). Les adaptations principales concernent le reshape spatial (tokens 1D → grille 2D/3D), le choix de baseline approprié au domaine médical, et la gestion des architectures hiérarchiques (Swin) ou hybrides (UNETR).

L'interprétabilité mécanistique (SAE, activation patching, circuit analysis) représente la **frontière de recherche la plus prometteuse** avec les outils ViT-Prisma et pyvene désormais disponibles. La validation clinique et l'intégration de connaissances médicales structurées (ontologies, concepts TCAV) constituent les défis majeurs pour le déploiement en pratique clinique.

Les bibliothèques **Captum**, **LXT**, **pytorch-grad-cam**, et **ViT-Prisma** fournissent une base d'implémentation solide couvrant l'ensemble des méthodes analysées, permettant une expérimentation rapide sur les datasets ISIC, chest X-ray et les benchmarks de segmentation médicale (BTCV, MSD).