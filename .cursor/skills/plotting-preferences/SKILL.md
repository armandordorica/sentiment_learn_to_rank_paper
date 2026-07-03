---
name: plotting-preferences
description: Apply project plotting preferences for charts and notebook visualizations. Use when creating, editing, or reviewing plots, charts, Jupyter visualizations, Plotly figures, Matplotlib figures, or data validation notebooks.
---

# Plotting Preferences

When extracting chart code into shared helpers, follow [refactoring](../refactoring/SKILL.md): keep viz functions domain-agnostic, follow PEP 8 and DRY, and leave dataset-specific framing to callers.

## Instructions

- Prefer Plotly plots for notebook and exploratory data visualizations.
- Avoid static Matplotlib plots unless there is a specific reason to generate a static artifact.
- Configure hover behavior to show only the exact point under the cursor.
- Do not use unified hover modes such as `hovermode="x unified"` when multiple series are present.
- Prefer `hovermode="closest"` for multi-series charts.
- Include useful hover fields such as identifiers, names, dates, and the plotted x/y values.
- Keep hover labels focused. Do not show all values across the chart at the same x-position.

## Plotly Defaults

For line charts:

```python
fig.update_traces(mode="lines+markers")
fig.update_layout(hovermode="closest")
```

For bar charts:

```python
fig.update_layout(hovermode="closest")
```

Use `hover_data` or `custom_data` to expose only the fields needed to inspect the selected point.
