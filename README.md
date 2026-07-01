# 🔓 Obliteración: apagarle el "no" a un LLM

### Interpretabilidad aplicada — álgebra lineal sobre las activaciones de un modelo real

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicobargioni/obliteracion-llm/blob/main/notebook.ipynb)

**Demo interactiva (sin instalar nada):** https://nicobargioni.github.io/obliteracion-llm/

En junio de 2026, Anthropic tuvo que frenar Fable 5 —su modelo más nuevo— durante casi
tres semanas, por orden del gobierno de EEUU, después de que alguien encontrara la forma
de esquivarle los filtros de seguridad. Este repo es la versión de laboratorio, transparente
y reproducible, del mismo tipo de mecanismo — corrida en un modelo chico y abierto
(`Qwen2.5-1.5B-Instruct`), con datos reales, no simulados.

> **Cómo usarlo:** abrí el badge **Open in Colab** de arriba → `Entorno de ejecución → Ejecutar todo`.
> Corre en CPU/T4 en pocos minutos, no necesita API key. O mirá la demo interactiva (el slider)
> directamente en GitHub Pages, sin ejecutar nada.

---

## Qué hay adentro

**La pregunta:** ¿el "rechazo" de un LLM (decir que no a un pedido) vive, aproximadamente, en
una única dirección lineal de su espacio de activaciones? Si es así, ¿alcanza con una resta
vectorial para apagarlo — sin reentrenar nada?

**El método (la técnica se llama *obliteración* / *abliteration*):**
1. Comparar las activaciones del modelo ante prompts que rechaza vs. prompts inocuos →
   la diferencia de medias es la **dirección de rechazo** (`r̂`).
2. Restar esa dirección de la salida de cada capa, durante la generación, con intensidad `α`
   (vía *forward hooks* — nada de tocar pesos permanentemente).
3. Barrer 15 valores de `α` y medir dos cosas: **tasa de rechazo** (8 prompts held-out) y
   **capacidad general** (perplejidad de control, sin relación con los prompts de prueba).

**El resultado real de esta corrida:**

| α | Rechazo (8 prompts held-out) | Capacidad |
|---|---|---|
| 0.0 (intacto) | 6/8 (75%) | 100% |
| 0.2 | 0/8 (0%) | 99.8% |
| 2.0 | 0/8 (0%) | 95.5% |
| 2.25 | 0/8 (0%) | **49.3%** (colapso) |

Hay una ventana amplia (`α` entre 0.2 y 2.0) donde el rechazo ya está apagado y el modelo
sigue siendo coherente. Pasado ese rango, la misma resta que apaga el rechazo rompe la
geometría que el resto de la red necesita — el modelo empieza a producir texto sin sentido.

---

## ⚠️ Nota de responsabilidad

Este repo mide *si el modelo rechaza o no* (una tasa, un booleano) — **no reproduce ni
publica contenido dañino generado**. Los prompts usados son de bajo riesgo real (cerraduras,
wifi, phishing genérico) y se eligieron así a propósito: alcanza para estudiar el mecanismo
sin tocar categorías de daño real (armas, CBRN, autolesión), que quedaron deliberadamente
afuera. El objetivo es entender y auditar mecanismos de seguridad de IA, no producir un
modelo sin filtro para uso real.

---

## Stack

- [`transformers`](https://huggingface.co/docs/transformers) + `torch` — `Qwen/Qwen2.5-1.5B-Instruct`
- Forward hooks para la intervención en tiempo de inferencia (sin fine-tuning)
- `plotly` para los gráficos (interactivos, en el notebook)
- Demo standalone (`index.html`) en vanilla JS/SVG, sin dependencias — pensada para GitHub Pages

El notebook se genera de forma reproducible con [`tools/build_notebook.py`](tools/build_notebook.py).

---

## Contenido del experimento más amplio

Este repo nació de una serie de posts/carrusel sobre el tema — más contexto (el caso Fable 5,
la comunidad OBLITERATUS en Hugging Face, el framework de severidad de jailbreaks) en el
post de LinkedIn vinculado desde el primer comentario.

---
*Nicolás Bargioni · Data Scientist · [nicolasbargioni.com](https://nicolasbargioni.com)*
