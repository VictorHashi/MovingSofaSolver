"""
Moving Sofa Problem — Algoritmo Genético
=========================================
Encontra a maior forma 2D que consegue passar por um corredor em L (largura 1)
usando um algoritmo genético.

AG:
  • População : 100 indivíduos
  • Genoma    : vetor de N raios em coordenadas polares (ângulos fixos)
  • Fitness   : área da forma (se navega pelo corredor) ou crédito parcial
  • Seleção   : roleta (proporcional ao fitness) + elitismo
  • Crossover : uniforme nos raios
  • Mutação   : gaussiana nos raios

Navegação:
  θ ∈ [0, π/2] — a forma entra horizontal e sai vertical, rotacionando 90°.
  Em cada passo θ, busca a posição (tx, ty) que maximiza o progresso.
  Critério de parada: bloqueio (sem posição válida) ou rotação completa.
  Critério de sucesso: centroide final com y ≥ 1 (saiu pelo corredor vertical).

Teclas:
  SPACE  — pausar / retomar evolução
  ENTER  — animar navegação do melhor indivíduo
  R      — toggle raios sensoriais
  I      — alternar entre modo único e por ilhas
  ESC    — sair
"""

import numpy as np
import cv2
import time

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

# ── Display ───────────────────────────────────────────────────────────────────
W, H     = 1200, 900
SCALE    = 200           # pixels por unidade de comprimento
ORIGIN_X = 550           # pixel da origem (0,0) do corredor
ORIGIN_Y = 750

# ── Algoritmo Genético ────────────────────────────────────────────────────────
POP_SIZE       = 200
N_GENES_SHAPE  = 32      # número de vértices (raios) por forma
N_GENES_ROT    = 250     # passos de rotação (trajeto) estendidos
N_GENES        = N_GENES_SHAPE + N_GENES_ROT
MUTATION_RATE  = 0.06     # probabilidade de mutar cada gene (reduzida para não destruir o trajeto)
MUTATION_SIGMA = 0.04     # desvio padrão da mutação gaussiana (raios)
CROSSOVER_RATE = 0.80     # probabilidade de crossover
N_ELITE        = 1        # quantos melhores preservar (elitismo aumentado)

# Modo de execução:
#   - "single"  : uma única população global
#   - "islands" : múltiplas subpopulações com migração periódica
RUN_MODE = "single"
ISLAND_COUNT = 4
MIGRATION_INTERVAL = 8
MIGRATION_RATE = 0.10

R_MIN, R_MAX = 0.05, 1.0    # limites dos raios no genoma (R_MAX=1.0 permite largura de até 2 unidades horizontais)

# ── Navegação ─────────────────────────────────────────────────────────────────
CLEARANCE_W    = 0.10     # peso da folga mínima no score

# ── Estado global ─────────────────────────────────────────────────────────────
show_rays = False
paused    = False

WINDOW = "Moving Sofa - GA | SPACE=Animar ENTER=Continuar R=raios I=single/ilhas ESC=sair"

# Tempo de execução acumulado e ilha do melhor candidato
execution_time = 0.0
best_island_idx = 0


# ══════════════════════════════════════════════════════════════════════════════
#  GEOMETRIA DO CORREDOR EM L
# ══════════════════════════════════════════════════════════════════════════════
#  Horizontal : y ∈ [0,1],  x ≤ 1
#  Vertical   : x ∈ [0,1],  y ≥ 0
#  Junção     : x ∈ [0,1],  y ∈ [0,1]

def w2p(pts):
    """Mundo (float) → pixel (int32), shape (N,2)."""
    p = np.asarray(pts, dtype=float).reshape(-1, 2)
    px = (ORIGIN_X + p[:, 0] * SCALE).astype(np.int32)
    py = (ORIGIN_Y - p[:, 1] * SCALE).astype(np.int32)
    return np.column_stack([px, py])


def _corridor_mask(points):
    """Máscara booleana para pontos no corredor em L."""
    x, y = points[:, 0], points[:, 1]
    horiz = (y >= 0) & (y <= 1) & (x >= -5.0) & (x <= 1)
    vert  = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 5.0)
    return (horiz | vert)

def in_corridor(pts):
    """True se o polígono inteiro está dentro do corredor em L.

    A checagem considera vértices e pontos intermediários de cada aresta,
    reduzindo o risco de uma aresta cruzar a parede do corredor.
    """
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    if len(pts) < 3:
        return False

    pts_closed = np.vstack([pts, pts[0]])
    edge_mid = 0.5 * (pts_closed[:-1] + pts_closed[1:])
    sample_points = np.vstack([pts, edge_mid])
    return bool(np.all(_corridor_mask(sample_points)))


def in_corridor_batch(candidates):
    """
    Testa múltiplas posições de uma vez.
    candidates : (K, N, 2) — K candidatos, N vértices cada.
    Retorna    : (K,) bool array.

    A validação usa os vértices e os pontos médios das arestas, de forma
    a capturar colisões envolvendo trechos do polígono, não só os vértices.
    """
    candidates = np.asarray(candidates, dtype=float)
    if candidates.ndim == 2:
        candidates = candidates[None, :, :]

    pts = candidates
    pts_next = np.roll(pts, -1, axis=1)
    edge_mid = 0.5 * (pts + pts_next)
    sample_points = np.concatenate([pts, edge_mid], axis=1)
    return _corridor_mask(sample_points.reshape(-1, 2)).reshape(len(candidates), -1).all(axis=1)


def rotate(pts, theta):
    c, s = np.cos(theta), np.sin(theta)
    return pts @ np.array([[c, -s], [s, c]]).T


def translate(pts, tx, ty):
    return pts + np.array([tx, ty])


# ══════════════════════════════════════════════════════════════════════════════
#  RAYCASTING — SENSORES DE DISTÂNCIA
# ══════════════════════════════════════════════════════════════════════════════

def compute_all_rays(shape_pts):
    """
    4 raios cardinais por vértice → distâncias às paredes.
    -1 = sem parede (raio sai da simulação).
    """
    x, y = shape_pts[:, 0], shape_pts[:, 1]
    in_h = (y >= 0) & (y <= 1) & (x <= 1)
    in_v = (x >= 0) & (x <= 1) & (y >= 0)
    inside = in_h | in_v

    d_right = np.where(inside, 1.0 - x, 0.0)
    d_down  = np.where(inside, y, 0.0)
    d_left  = np.where(inside, np.where(y > 1.0, x, -1.0), 0.0)
    d_up    = np.where(inside, np.where(x < 0.0, 1.0 - y, -1.0), 0.0)

    rays = np.column_stack([d_right, d_left, d_up, d_down])
    finite = rays[rays >= 0]
    min_clearance = float(finite.min()) if finite.size > 0 else 0.0
    return rays, min_clearance


def _ray_color(d):
    """Verde (longe) → Vermelho (perto)."""
    t = float(np.clip(d / 0.8, 0.0, 1.0))
    return (0, int(200 * t), int(200 * (1 - t)))


# ══════════════════════════════════════════════════════════════════════════════
#  REPRESENTAÇÃO DO GENOMA
# ══════════════════════════════════════════════════════════════════════════════

def genome_to_shape(radii):
    """
    Converte genoma (vetor de raios polares) em vértices da forma.
    Normaliza para que a dimensão mais estreita ≤ 0.95 (cabe no corredor).
    """
    n = len(radii)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    x -= x.mean()
    y -= y.mean()

    # Escalar: dimensão mínima → 0.95 (largura do corredor)
    x_ext = x.max() - x.min() + 1e-9
    y_ext = y.max() - y.min() + 1e-9
    min_ext = min(x_ext, y_ext)
    factor = 0.95 / min_ext

    # Limitar dimensão máxima a 4.0 (evita formas absurdas)
    max_ext = max(x_ext, y_ext)
    if max_ext * factor > 4.0:
        factor = 4.0 / max_ext

    x *= factor
    y *= factor
    return np.column_stack([x, y])


def shape_area(pts):
    """Área via fórmula do cadarço (Shoelace)."""
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


# ══════════════════════════════════════════════════════════════════════════════
#  NAVEGAÇÃO — TRAJETÓRIA PELO CORREDOR EM L
# ══════════════════════════════════════════════════════════════════════════════

def navigate(shape, rots, use_clearance=False):
    """
    Tenta navegar a forma pelo corredor em L de x=-1 até y=2.

    A forma translada de maneira gulosa e rotaciona conforme o genoma!

    Retorna (frames, progress, success):
      frames   — lista de arrays (N,2) em cada passo
      progress — fração do percurso completada ∈ [0, 1]
      success  — True se todos os pontos atingiram y >= 2.0
    """
    y_ext = shape[:, 1].max() - shape[:, 1].min()
    if y_ext > 0.98:
        return [], 0.0, False

    # Posição inicial: frente do sofá em x = -1.0
    tx = -1.0 - shape[:, 0].max()
    ty = 0.5 - (shape[:, 1].max() + shape[:, 1].min()) / 2.0
    theta = 0.0

    if not in_corridor(translate(shape, tx, ty)):
        for try_ty in np.linspace(0.1, 0.9, 10):
            if in_corridor(translate(shape, tx, try_ty)):
                ty = try_ty
                break
        else:
            return [], 0.0, False

    frames = []
    
    dtx_opts = np.linspace(-0.05, 0.20, 6)
    dty_opts = np.linspace(-0.05, 0.20, 6)
    DTX, DTY = np.meshgrid(dtx_opts, dty_opts)
    base_offsets = np.column_stack([DTX.ravel(), DTY.ravel()])

    max_steps = len(rots)

    for step in range(max_steps):
        rot = rotate(shape, theta)
        current_pos = translate(rot, tx, ty)
        frames.append(current_pos.copy())
        
        # Sucesso: todos os pontos passaram da marca de 1 unidade após o vértice (y = 2.0)
        if current_pos[:, 1].min() >= 2.0:
            return frames, 1.0, True

        dth = rots[step]
        if theta + dth > np.pi / 2:
            dth = max(0.0, np.pi / 2 - theta)
            
        nth = theta + dth
        rot_cand = rotate(shape, nth)
        
        all_tx = tx + base_offsets[:, 0]
        all_ty = ty + base_offsets[:, 1]
        
        candidates = rot_cand[None, :, :] + np.column_stack([all_tx, all_ty])[:, None, :]
        
        valid_mask = in_corridor_batch(candidates)
        valid_idx = np.where(valid_mask)[0]
        
        if len(valid_idx) == 0:
            break
            
        v_tx = all_tx[valid_idx]
        v_ty = all_ty[valid_idx]
        
        # Como a rotação é ditada pelo genoma, pontuamos apenas o avanço guloso
        scores = v_tx + v_ty
        
        if use_clearance:
            for j, vi in enumerate(valid_idx):
                _, mc = compute_all_rays(candidates[vi])
                scores[j] += CLEARANCE_W * mc

        max_idx = int(np.argmax(scores))
        ntx, nty = v_tx[max_idx], v_ty[max_idx]
        
        if ntx <= tx + 1e-4 and nty <= ty + 1e-4 and nth <= theta + 1e-4:
            break
            
        tx, ty, theta = ntx, nty, nth

    progress = max(0.0, (tx + 1.0 + ty) / 4.0)
    return frames, min(0.99, progress), False


# ══════════════════════════════════════════════════════════════════════════════
#  FITNESS
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_fitness(genome):
    """
    Fitness = área da forma × fator de navegação.
      • Navegou com sucesso : fitness = área
      • Bloqueado           : fitness = área × progresso^2
    """
    shape = genome_to_shape(genome[:N_GENES_SHAPE])
    rots = genome[N_GENES_SHAPE:]
    area = shape_area(shape)
    _, progress, success = navigate(shape, rots, use_clearance=False)

    if success:
        return area * 5.0  # Bônus significativo por completar o trajeto
    else:
        # A penalidade progress^2 permite que formas maiores que empacam no final 
        # superem formas quadradas minúsculas que terminam rápido.
        return area * (progress ** 2)


# ══════════════════════════════════════════════════════════════════════════════
#  OPERAÇÕES GENÉTICAS
# ══════════════════════════════════════════════════════════════════════════════

def random_genome():
    """Genoma aleatório com raios moderados."""
    return np.random.uniform(R_MIN + 0.05, R_MAX * 0.65, N_GENES)


def roulette_selection(population, fitnesses):
    """Seleção por roleta: probabilidade proporcional ao fitness."""
    total = fitnesses.sum()
    if total <= 0:
        return population[np.random.randint(len(population))].copy()
    probs = fitnesses / total
    idx = np.random.choice(len(population), p=probs)
    return population[idx].copy()


def crossover(p1, p2):
    """Crossover misto: uniforme na forma, 1-point no trajeto."""
    c1, c2 = p1.copy(), p2.copy()
    
    # Uniforme para a forma (genes independentes)
    mask = np.random.random(N_GENES_SHAPE) < 0.5
    c1[:N_GENES_SHAPE] = np.where(mask, p1[:N_GENES_SHAPE], p2[:N_GENES_SHAPE])
    c2[:N_GENES_SHAPE] = np.where(mask, p2[:N_GENES_SHAPE], p1[:N_GENES_SHAPE])
    
    # 1-point para o trajeto (mantém a continuidade temporal da rotação)
    if np.random.rand() < CROSSOVER_RATE:
        pt = np.random.randint(N_GENES_SHAPE, N_GENES)
        c1[pt:], c2[pt:] = p2[pt:].copy(), p1[pt:].copy()
        
    return c1, c2


def create_population(size):
    pop = []
    thetas = np.linspace(0, 2*np.pi, N_GENES_SHAPE, endpoint=False)
    # Limita o raio inicial para que a projeção y não exceda [-0.48, 0.48]
    # Isso garante que a primeira geração já caiba no corredor horizontal
    # e permite alta variabilidade na largura (x)!
    max_r = np.abs(0.48 / (np.sin(thetas) + 1e-5))
    max_r = np.clip(max_r, R_MIN, R_MAX)
    
    for _ in range(size):
        radii = np.random.uniform(R_MIN, max_r)
        rots  = np.random.uniform(0.0, 0.02, N_GENES_ROT)
        pop.append(np.concatenate([radii, rots]))
    return pop


def create_island_populations(total_size, island_count):
    """Cria múltiplas subpopulações com tamanho equilibrado."""
    base = total_size // island_count
    sizes = [base] * island_count
    sizes[-1] += total_size - base * island_count
    return [create_population(max(20, size)) for size in sizes]


def evaluate_population(population):
    """Avalia uma população inteira e retorna os fitnesses."""
    return np.array([evaluate_fitness(g) for g in population], dtype=float)


def evolve_population(population):
    """Realiza uma geração de evolução para uma única população."""
    fitnesses = evaluate_population(population)
    best_idx = int(np.argmax(fitnesses))
    best_fitness = float(fitnesses[best_idx])
    avg_fitness = float(fitnesses.mean())

    sorted_idx = np.argsort(fitnesses)[::-1]
    new_pop = []

    for i in range(min(N_ELITE, len(sorted_idx))):
        new_pop.append(population[sorted_idx[i]].copy())

    while len(new_pop) < len(population):
        p1 = roulette_selection(population, fitnesses)
        p2 = roulette_selection(population, fitnesses)

        if np.random.random() < CROSSOVER_RATE:
            c1, c2 = crossover(p1, p2)
        else:
            c1, c2 = p1.copy(), p2.copy()

        new_pop.append(mutate(c1))
        if len(new_pop) < len(population):
            new_pop.append(mutate(c2))

    return new_pop, population[best_idx].copy(), best_fitness, avg_fitness


def initialize_population():
    """Inicializa a população conforme o modo atual."""
    if RUN_MODE == "islands":
        return create_island_populations(POP_SIZE, ISLAND_COUNT)
    return create_population(POP_SIZE)


def get_display_shape(population, best_genome=None):
    """Retorna um genoma para renderização inicial."""
    if best_genome is not None:
        return genome_to_shape(best_genome[:N_GENES_SHAPE])
    if RUN_MODE == "islands" and population and population[0]:
        return genome_to_shape(population[0][0][:N_GENES_SHAPE])
    if population:
        return genome_to_shape(population[0][:N_GENES_SHAPE])
    return None


def migrate_between_islands(islands):
    """Migra um indivíduo entre ilhas com periodicidade."""
    if len(islands) < 2:
        return islands

    island_fits = []
    for island in islands:
        fits = evaluate_population(island)
        island_fits.append(float(np.max(fits)))

    donor_idx = int(np.argmax(island_fits))
    recipient_idx = donor_idx
    while recipient_idx == donor_idx:
        recipient_idx = int(np.random.randint(len(islands)))

    donor_island = islands[donor_idx]
    donor_fits = evaluate_population(donor_island)
    donor_best_idx = int(np.argmax(donor_fits))
    migrant = donor_island[donor_best_idx].copy()

    target_island = islands[recipient_idx]
    target_idx = int(np.random.randint(len(target_island)))
    target_island[target_idx] = migrant
    return islands


def mutate(genome):
    for i in range(N_GENES):
        if i < N_GENES_SHAPE:
            # Mutação de 5% para a forma (maior variabilidade estrutural)
            if np.random.rand() < 0.05:
                genome[i] += np.random.randn() * MUTATION_SIGMA
                genome[i] = np.clip(genome[i], R_MIN, R_MAX)
        else:
            # Mutação de 2% para o trajeto (preserva melhor o movimento aprendido)
            if np.random.rand() < 0.02:
                genome[i] += np.random.randn() * 0.015
                genome[i] = np.clip(genome[i], -0.05, 0.10)
    return genome


# ══════════════════════════════════════════════════════════════════════════════
#  DESENHO
# ══════════════════════════════════════════════════════════════════════════════

def draw_corridor(canvas):
    """Desenha corredor em L com grade, paredes e marcadores."""
    # Grade
    for v in np.arange(-4.5, 2.0, 0.5):
        p1, p2 = w2p([[v, -0.5]])[0], w2p([[v, 4.5]])[0]
        cv2.line(canvas, tuple(p1), tuple(p2), (28, 28, 28), 1)
    for u in np.arange(-0.5, 4.5, 0.5):
        p1, p2 = w2p([[-4.5, u]])[0], w2p([[1.5, u]])[0]
        cv2.line(canvas, tuple(p1), tuple(p2), (28, 28, 28), 1)

    # Área livre
    horiz = np.array([[-4.5, 0], [1, 0], [1, 1], [-4.5, 1]])
    vert  = np.array([[0, 0], [1, 0], [1, 4.0], [0, 4.0]])
    cv2.fillPoly(canvas, [w2p(horiz)], (25, 30, 40))
    cv2.fillPoly(canvas, [w2p(vert)],  (25, 30, 40))

    # Paredes
    wc, wt = (90, 95, 110), 2
    cv2.line(canvas, tuple(w2p([[-4.5, 0]])[0]), tuple(w2p([[1, 0]])[0]),   wc, wt)
    cv2.line(canvas, tuple(w2p([[-4.5, 1]])[0]), tuple(w2p([[0, 1]])[0]),   wc, wt)
    cv2.line(canvas, tuple(w2p([[0, 1]])[0]),     tuple(w2p([[0, 4.0]])[0]), wc, wt)
    cv2.line(canvas, tuple(w2p([[1, 0]])[0]),     tuple(w2p([[1, 4.0]])[0]), wc, wt)

    # Linhas de entrada e saída (1 unidade do vértice)
    wc_dash = (60, 140, 90)
    cv2.line(canvas, tuple(w2p([[-1.0, 0]])[0]), tuple(w2p([[-1.0, 1]])[0]), wc_dash, 1)
    cv2.line(canvas, tuple(w2p([[0, 2.0]])[0]), tuple(w2p([[1.0, 2.0]])[0]), wc_dash, 1)

    # Marcadores de unidade
    for v in [-4, -3, -2, -1, 0, 1]:
        p = w2p([[v, -0.08]])[0]
        cv2.putText(canvas, str(v), tuple(p), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (55, 55, 65), 1, cv2.LINE_AA)
    for u in [0, 1, 2, 3, 4]:
        p = w2p([[-0.15, u]])[0]
        cv2.putText(canvas, str(u), tuple(p), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (55, 55, 65), 1, cv2.LINE_AA)


def draw_shape(canvas, pts):
    """Desenha a forma (fill + outline + vértices + raios opcionais)."""
    if pts is None:
        return
    ok = in_corridor(pts)
    fill_c = (50, 150, 95) if ok else (150, 50, 50)
    edge_c = (110, 210, 150) if ok else (210, 110, 110)

    pts_px = w2p(pts)
    cv2.fillPoly(canvas, [pts_px], fill_c)
    cv2.polylines(canvas, [pts_px], True, edge_c, 2)

    dot_r = max(2, min(4, 200 // max(1, len(pts))))
    for p in pts_px:
        cv2.circle(canvas, tuple(p), dot_r, (0, 200, 230), -1)

    # Raios sensoriais
    if show_rays:
        rays, _ = compute_all_rays(pts)
        for i, pt in enumerate(pts):
            px_pt = w2p([[pt[0], pt[1]]])[0]
            d_r, d_l, d_u, d_d = rays[i]
            dirs = [
                (d_r, pt[0] + d_r, pt[1]),
                (d_l, pt[0] - d_l, pt[1]),
                (d_u, pt[0],       pt[1] + d_u),
                (d_d, pt[0],       pt[1] - d_d),
            ]
            for d, ex, ey in dirs:
                if d >= 0:
                    end = w2p([[ex, ey]])[0]
                    cv2.line(canvas, tuple(px_pt), tuple(end),
                             _ray_color(d), 1, cv2.LINE_AA)


def draw_stats(canvas, gen, best_fit, avg_fit, best_ever_fit,
               best_ever_gen, pass_rate, elapsed, status,
               best_island_idx_local=None):
    """Painel de estatísticas do AG no lado direito."""
    x0 = W - 390
    y0 = 35
    dy = 25

    cv2.putText(canvas, "ALGORITMO GENETICO", (x0, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 185, 255), 2, cv2.LINE_AA)

    mode_label = "ILHAS" if RUN_MODE == "islands" else "UNICA"
    if RUN_MODE == "islands" and best_island_idx_local is not None:
        best_general_row = f"    (geracao {best_ever_gen}, ilha {best_island_idx_local + 1})"
    else:
        best_general_row = f"    (geracao {best_ever_gen})"
    rows = [
        "",
        f"Geracao       : {gen}",
        f"Populacao     : {POP_SIZE}",
        f"Vertices      : {N_GENES}",
        f"Modo          : {mode_label}",
        f"Ilhas         : {ISLAND_COUNT}" if RUN_MODE == "islands" else "",
        f"Migra. a cada : {MIGRATION_INTERVAL}" if RUN_MODE == "islands" else "",
        f"Mut. rate     : {MUTATION_RATE}",
        f"Crossover     : {CROSSOVER_RATE}",
        "",
        f"Melhor fitness: {best_fit:.6f}",
        f"Media fitness : {avg_fit:.6f}",
        f"Taxa sucesso  : {pass_rate:.0%}",
        "",
        f">>> MELHOR GERAL: {best_ever_fit:.6f}",
        best_general_row,
        "",
        f"Tempo total   : {execution_time:.1f}s",
        f"Status        : {status}",
        "",
        f"Raios         : {'ON' if show_rays else 'OFF'}  (R)",
    ]

    for i, txt in enumerate(rows):
        if not txt:
            continue
        y = y0 + (i + 1) * dy
        if txt.startswith(">>>"):
            col = (80, 255, 180)
        elif "Status" in txt:
            col = (200, 200, 140)
        else:
            col = (160, 165, 160)
        cv2.putText(canvas, txt, (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)

    # Controles
    cy = H - 300
    for i, txt in enumerate([
        "SPACE  pausar/retomar",
        "ENTER  animar melhor",
        "R      raios ON/OFF",
        "I      single/ilhas",
        "ESC    sair",
    ]):
        cv2.putText(canvas, txt, (x0, cy + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (75, 80, 75), 1, cv2.LINE_AA)


def make_frame(shape_pts, gen, best_fit, avg_fit, best_ever_fit,
               best_ever_gen, pass_rate, elapsed, status,
               best_island_idx_local=None):
    """Frame completo: corredor + forma + stats."""
    if best_island_idx_local is None:
        best_island_idx_local = best_island_idx

    canvas = np.full((H, W, 3), 18, dtype=np.uint8)
    draw_corridor(canvas)
    draw_shape(canvas, shape_pts)

    if shape_pts is not None:
        area = shape_area(shape_pts)
        cx = float(shape_pts[:, 0].mean())
        cy = float(shape_pts[:, 1].mean())
        p = w2p([[cx, cy]])[0]
        cv2.putText(canvas, f"A={area:.4f}", (int(p[0]) - 30, int(p[1]) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1,
                    cv2.LINE_AA)

    draw_stats(canvas, gen, best_fit, avg_fit, best_ever_fit,
               best_ever_gen, pass_rate, elapsed, status,
               best_island_idx_local)
    return canvas


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — LOOP DO ALGORITMO GENÉTICO
# ══════════════════════════════════════════════════════════════════════════════

cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW, W, H)

# ── Inicializar população ─────────────────────────────────────────────────────
population = initialize_population()
generation    = 0
best_ever_fit = 0.0
best_ever_genome = None
best_ever_gen = 0
anim_frames_gen = -1
paused_frames = []
anim_idx = 0

# Variáveis de tracking (evita erros de referência)
best_fitness = 0.0
avg_fitness  = 0.0
pass_rate    = 0.0
elapsed      = 0.0
best_idx     = 0

running = True
while running:
    # ══════════════════════════════════════════════════════════════════════
    #  EVOLUÇÃO (quando não pausado)
    # ══════════════════════════════════════════════════════════════════════
    if not paused:
        # ── Mostrar status de avaliação ──────────────────────────────────
        status_text = f"Avaliando geracao {generation}..."
        init_shape = get_display_shape(population)
        if init_shape is not None:
            placed = translate(init_shape, -1.0 - init_shape[:, 0].max(), 0.5)
        else:
            placed = None
        frame = make_frame(placed, generation, best_fitness, avg_fitness,
                           best_ever_fit, best_ever_gen, pass_rate,
                           elapsed, status_text, best_island_idx)
        cv2.imshow(WINDOW, frame)
        cv2.waitKey(1)

        # ── Avaliar fitness de toda a população ──────────────────────────
        t0 = time.time()
        if RUN_MODE == "islands":
            evolved_islands = []
            island_best_fits = []
            island_best_genomes = []
            island_avg_fits = []

            for island in population:
                new_island, best_genome, best_fit, avg_fit = evolve_population(island)
                evolved_islands.append(new_island)
                island_best_fits.append(best_fit)
                island_best_genomes.append(best_genome)
                island_avg_fits.append(avg_fit)

            if generation > 0 and generation % MIGRATION_INTERVAL == 0:
                evolved_islands = migrate_between_islands(evolved_islands)

            population = evolved_islands
            best_idx = int(np.argmax(island_best_fits))
            best_island_idx = best_idx
            best_fitness = float(island_best_fits[best_idx])
            avg_fitness = float(np.mean(island_avg_fits))
            best_genome = island_best_genomes[best_idx]
            pass_rate = float(np.mean([fit > 0.0 for fit in island_best_fits]))
        else:
            fitnesses = evaluate_population(population)
            best_idx = int(np.argmax(fitnesses))
            best_fitness = float(fitnesses[best_idx])
            avg_fitness = float(fitnesses.mean())
            best_genome = population[best_idx].copy()
            pass_rate = float(np.mean(fitnesses > 0.0))
            best_island_idx = 0
            population, best_genome, best_fitness, avg_fitness = evolve_population(population)

        elapsed = time.time() - t0
        execution_time += elapsed

        # Atualizar melhor geral
        if best_fitness > best_ever_fit:
            best_ever_fit    = best_fitness
            best_ever_genome = best_genome.copy()
            best_ever_gen    = generation

        # ── Mostrar melhor da geração ────────────────────────────────────
        best_shape = genome_to_shape(best_genome[:N_GENES_SHAPE])
        placed = translate(best_shape, -1.0 - best_shape[:, 0].max(), 0.5)
        status_text = "Evoluindo..."
        frame = make_frame(placed, generation, best_fitness, avg_fitness,
                           best_ever_fit, best_ever_gen, pass_rate,
                           elapsed, status_text, best_island_idx)
        cv2.imshow(WINDOW, frame)

        generation += 1

    else:
        # ── Pausado — mostrar melhor geral animando ──────────────────────
        if best_ever_genome is not None:
            shape = genome_to_shape(best_ever_genome[:N_GENES_SHAPE])
            rots = best_ever_genome[N_GENES_SHAPE:]
            
            if anim_frames_gen != best_ever_gen:
                paused_frames, _, _ = navigate(shape, rots, use_clearance=False)
                anim_frames_gen = best_ever_gen
                anim_idx = 0
                
            if len(paused_frames) > 0:
                placed = paused_frames[anim_idx]
                anim_idx = (anim_idx + 1) % len(paused_frames)
                time.sleep(0.03) # suaviza a velocidade da animacao do loop
            else:
                placed = translate(shape, -1.0 - shape[:, 0].max(), 0.5)
        else:
            placed = None
            
        frame = make_frame(placed, generation, best_fitness, avg_fitness,
                           best_ever_fit, best_ever_gen, pass_rate,
                           elapsed, "Pausado (Animando Trajeto)", best_island_idx)
        cv2.imshow(WINDOW, frame)

    # ══════════════════════════════════════════════════════════════════════
    #  TECLAS
    # ══════════════════════════════════════════════════════════════════════
    wait_ms = 30 if not paused else 100
    key = cv2.waitKey(wait_ms) & 0xFF

    if key == 27:                     # ESC — sair
        running = False

    elif key == 32:                   # SPACE — pausar/retomar
        paused = not paused

    elif key == ord('r'):             # R — toggle raios
        show_rays = not show_rays

    elif key == ord('i'):             # I — alternar entre single e ilhas
        RUN_MODE = "islands" if RUN_MODE != "islands" else "single"
        population = initialize_population()
        generation = 0
        best_ever_fit = 0.0
        best_ever_genome = None
        best_ever_gen = 0
        anim_frames_gen = -1
        paused_frames = []
        anim_idx = 0
        best_fitness = 0.0
        avg_fitness = 0.0
        pass_rate = 0.0
        elapsed = 0.0
        best_idx = 0
        best_island_idx = 0
        execution_time = 0.0

    elif key == 13:                   # ENTER — animar melhor
        target = best_ever_genome
        if target is not None:
            shape = genome_to_shape(target[:N_GENES_SHAPE])
            rots = target[N_GENES_SHAPE:]
            frames_anim, prog, success = navigate(
                shape, rots, use_clearance=True)

            for pts in frames_anim:
                af = make_frame(pts, generation, best_fitness, avg_fitness,
                                best_ever_fit, best_ever_gen, pass_rate,
                                elapsed, "Animando melhor...", best_island_idx)
                cv2.imshow(WINDOW, af)
                k = cv2.waitKey(16) & 0xFF
                if k == 27:
                    running = False
                    break

            if running and frames_anim:
                result = ("\u2713 Passou!" if success
                          else f"\u2717 Bloqueado ({prog:.0%})")
                af = make_frame(frames_anim[-1], generation,
                                best_fitness, avg_fitness,
                                best_ever_fit, best_ever_gen,
                                pass_rate, elapsed, result, best_island_idx)
                cv2.imshow(WINDOW, af)
                cv2.waitKey(1500)

cv2.destroyAllWindows()