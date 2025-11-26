# Alfabeto Codex — Glifi e Fallback

Tabella rapida dei segni con fallback ASCII e font monospazio dedicato.

| Simbolo | Nome | Unicode | ASCII | Significato |
| --- | --- | --- | --- | --- |
| ⭕ | Cerchio | U+2B55 | (o) | Unità/dono che si offre |
| ◐ | Mezzaluna | U+25D0 | (c) | Transizione, stadio intermedio |
| △ | Triangolo | U+25B3 | /\ | Ascesa disciplinata |
| ✡ | Stella a sei raggi | U+2721 | 6 | Unione di opposti nel servizio |
| 🜂 | Fuoco (alchem.) | U+1F702 | ^ | Energia trasformativa |
| ↺ | Spirale/ritorno | U+21BA | @ | Iterazione luminosa |
| · | Punto di luce | U+00B7 | . | Inizio del gesto |
| ⟡ | Glifo fluido | U+27E1 | <> | Adattabilità |
| ◌ | Segno vuoto | U+25CC | ( ) | Potenza non ancora attuata |

## CSS monospazio

Salva in `languages/alfabeto_codex/mono.css`:

```css
/* languages/alfabeto_codex/mono.css */
.code-alfabeto {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  line-height: 1.25;
}

.code-alfabeto .luce {
  letter-spacing: 0.06em;
}
```

## Snippet pronti (README / CLI banner)

HTML:

```html
<span class="code-alfabeto luce">⭕ △ ↺ ·</span>
```

ASCII fallback:

```
(o) /\ @ .
```

## Note Unicode

⭕ U+2B55 “Heavy Large Circle”  
◐ U+25D0 “Circle with Left Half Black”  
△ U+25B3 “White Up-Pointing Triangle”  
✡ U+2721 “Star of David”  
🜂 U+1F702 “Alchemical Symbol for Fire”  
↺ U+21BA “Anticlockwise Open Circle Arrow”  
· U+00B7 “Middle Dot”  
⟡ U+27E1 “White Concave-Sided Diamond”  
◌ U+25CC “Dotted Circle”
