document.querySelectorAll("[data-sort-table]").forEach(table => {
  table.querySelectorAll("th").forEach((th, column) => {
    th.addEventListener("click", () => {
      const body = table.tBodies[0];
      const rows = Array.from(body.rows);
      const ascending = th.dataset.direction !== "asc";
      table.querySelectorAll("th").forEach(x => delete x.dataset.direction);
      th.dataset.direction = ascending ? "asc" : "desc";
      rows.sort((a, b) => {
        const av = a.cells[column]?.dataset.value ?? a.cells[column]?.innerText ?? "";
        const bv = b.cells[column]?.dataset.value ?? b.cells[column]?.innerText ?? "";
        const an = Number(av), bn = Number(bv);
        const result = Number.isFinite(an) && Number.isFinite(bn)
          ? an - bn : av.localeCompare(bv);
        return ascending ? result : -result;
      });
      rows.forEach(row => body.appendChild(row));
    });
  });
});
document.querySelectorAll("[data-split-button]").forEach(button => {
  button.addEventListener("click", () => {
    const split = button.dataset.splitButton;
    document.querySelectorAll("[data-split-panel]").forEach(panel => {
      panel.hidden = panel.dataset.splitPanel !== split;
    });
    document.querySelectorAll("[data-split-button]").forEach(other => {
      other.classList.toggle("active", other === button);
    });
  });
});
