const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Footer, AlignmentType, LevelFormat, ImageRun,
  BorderStyle, WidthType, ShadingType, VerticalAlign, PageNumber, PageBreak
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const FONT_HEI = "\u9ED1\u4F53";
const FONT_SONG = "\u5B8B\u4F53";
const FONT_KAI = "\u6977\u4F53";

const bdr = { style: BorderStyle.SINGLE, size: 1, color: "000000" };
const bdrs = { top: bdr, bottom: bdr, left: bdr, right: bdr };
const cMar = { top: 57, bottom: 57, left: 57, right: 57 };

function sr(text, o = {}) {
  return new TextRun({ text, font: { name: FONT_SONG, eastAsia: FONT_SONG }, size: 24, ...o });
}
function sb(text, o = {}) { return sr(text, { bold: true, ...o }); }
function hr(text, o = {}) {
  return new TextRun({ text, font: { name: FONT_HEI, eastAsia: FONT_HEI }, size: 24, ...o });
}
function kr(text, o = {}) {
  return new TextRun({ text, font: { name: FONT_KAI, eastAsia: FONT_KAI }, size: 24, ...o });
}

function bp(runs, o = {}) {
  return new Paragraph({ spacing: { line: 360, lineRule: "auto" }, indent: { firstLine: 480 }, ...o, children: Array.isArray(runs) ? runs : [runs] });
}

function hCell(t, w) {
  return new TableCell({
    width: { size: w, type: WidthType.DXA }, borders: bdrs, margins: cMar,
    shading: { fill: "D9E2F3", type: ShadingType.CLEAR }, verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [sb(t, { size: 20 })] })]
  });
}
function dCell(t, w, bold) {
  return new TableCell({
    width: { size: w, type: WidthType.DXA }, borders: bdrs, margins: cMar, verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [bold ? sb(t, { size: 20 }) : sr(t, { size: 20 })] })]
  });
}

function makeTable(headers, rows, colWidths) {
  const hRow = new TableRow({ children: headers.map((h, i) => hCell(h, colWidths[i])) });
  const dRows = rows.map(r => new TableRow({
    children: r.vals.map((v, i) => dCell(v, colWidths[i], r.bold))
  }));
  const tw = colWidths.reduce((a, b) => a + b, 0);
  return new Table({ width: { size: tw, type: WidthType.DXA }, rows: [hRow, ...dRows] });
}

function imgPara(filePath, w, h, cap) {
  const kids = [];
  try {
    const data = fs.readFileSync(filePath);
    kids.push(new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new ImageRun({ type: "png", data, transformation: { width: w, height: h },
        altText: { title: cap, description: cap, name: cap } })]
    }));
  } catch (e) {
    kids.push(new Paragraph({ children: [sr("[" + cap + " - \u56FE\u7247\u672A\u627E\u5230]")] }));
  }
  kids.push(new Paragraph({
    spacing: { before: 60, after: 120 }, alignment: AlignmentType.CENTER,
    children: [sr(cap, { size: 21 })]
  }));
  return kids;
}

function coverField(label, value) {
  return new Paragraph({
    spacing: { after: 360 }, tabStops: [{ type: "left", position: 1276 }],
    children: [
      new TextRun({ font: { name: FONT_HEI, eastAsia: FONT_HEI }, size: 30, children: ["\t"] }),
      hr(label, { size: 30 }), hr(value, { size: 30, underline: { type: "single" } })
    ]
  });
}

// ===== COVER =====
const cover = [
  new Paragraph({ spacing: { before: 960 }, alignment: AlignmentType.CENTER,
    children: [hr("\u7535 \u5B50 \u79D1 \u6280 \u5927 \u5B66", { size: 56 })] }),
  new Paragraph({ spacing: { before: 240, after: 1680 }, alignment: AlignmentType.CENTER,
    children: [hr("\u4E13\u4E1A\u5B66\u4F4D\u7814\u7A76\u751F\u5B66\u4F4D\u8BBA\u6587\u4E2D\u671F\u8003\u8BC4\u8868", { size: 46, characterSpacing: 50 })] }),
  coverField("\u653B\u8BFB\u5B66\u4F4D\u7EA7\u522B\uFF1A ", "\u25A1\u535A\u58EB   \u2611\u7855\u58EB"),
  coverField("\u57F9\u517B\u65B9\u5F0F\uFF1A     ", "\u2611\u5168\u65E5\u5236    \u25A1\u975E\u5168\u65E5\u5236"),
  coverField("\u4E13\u4E1A\u5B66\u4F4D\u7C7B\u522B\u53CA\u9886\u57DF\uFF1A", "   \u8F6F\u4EF6\u5DE5\u7A0B          "),
  coverField("\u5B66        \u9662\uFF1A", "   \u4FE1\u606F\u4E0E\u8F6F\u4EF6\u5DE5\u7A0B\u5B66\u9662      "),
  coverField("\u5B66        \u53F7\uFF1A", "        202421090XXX       "),
  coverField("\u59D3        \u540D\uFF1A", "         \u5218\u4FCA\u5E0C            "),
  new Paragraph({ spacing: { after: 360 }, tabStops: [{ type: "left", position: 1276 }],
    children: [
      new TextRun({ font: { name: FONT_HEI, eastAsia: FONT_HEI }, size: 30, children: ["\t"] }),
      hr("\u8BBA\u6587\u9898\u76EE\uFF1A", { size: 30 }),
      hr("\u57FA\u4E8E\u56E0\u679C\u8868\u5F81\u5B66\u4E60\u7684\u80BA\u764C\u9884\u9632\u63A7\u5236", { size: 30, underline: { type: "single" } })
    ] }),
  new Paragraph({ spacing: { after: 360 }, tabStops: [{ type: "left", position: 1276 }],
    children: [
      new TextRun({ font: { name: FONT_HEI, eastAsia: FONT_HEI }, size: 30, children: ["\t"] }),
      hr("              ", { size: 30 }),
      hr("\u8F85\u52A9\u51B3\u7B56\u7814\u7A76\u4E0E\u5B9E\u73B0", { size: 30, underline: { type: "single" } }),
      hr("                     ", { size: 30, underline: { type: "single" } })
    ] }),
  coverField("\u6821\u5185\u6307\u5BFC\u6559\u5E08\uFF1A", "         XXX              "),
  coverField("\u6821\u5916\u6307\u5BFC\u6559\u5E08\uFF1A", "         XXX              "),
  new Paragraph({ spacing: { after: 360 }, tabStops: [{ type: "left", position: 1276 }],
    children: [
      new TextRun({ font: { name: FONT_HEI, eastAsia: FONT_HEI }, size: 30, children: ["\t"] }),
      hr("\u586B\u8868\u65E5\u671F\uFF1A", { size: 30 }),
      hr("  2026 ", { size: 30, underline: { type: "single" } }),
      hr("\u5E74", { size: 30 }),
      hr("   3 ", { size: 30, underline: { type: "single" } }),
      hr("\u6708", { size: 30 }),
      hr("  24  ", { size: 30, underline: { type: "single" } }),
      hr("\u65E5", { size: 30 })
    ] }),
  new Paragraph({ spacing: { before: 1200 }, alignment: AlignmentType.CENTER,
    children: [kr("\u7535\u5B50\u79D1\u6280\u5927\u5B66\u7814\u7A76\u751F\u9662", { size: 32 })] })
];

// ===== CONTENT BUILDER =====
const content = require("./gen_content.js")(ROOT, { sr, sb, hr, kr, bp, makeTable, imgPara, Paragraph, TextRun, AlignmentType, FONT_SONG });

// Main table row for section 3
const TW = 9241;
const C1 = Math.round(TW * 0.44);
const C2 = TW - C1;

const mainTable = new Table({
  width: { size: 5000, type: WidthType.PERCENTAGE },
  rows: [
    new TableRow({ children: [new TableCell({ width: { size: 5000, type: WidthType.PERCENTAGE }, columnSpan: 2, borders: bdrs, margins: cMar, verticalAlign: VerticalAlign.CENTER,
      children: [new Paragraph({ children: [sb("1.\u5F00\u9898\u62A5\u544A\u901A\u8FC7\u65F6\u95F4\uFF1A"), sr("  2025  \u5E74  11  \u6708  28  \u65E5", { underline: { type: "single" } })] })] })] }),
    new TableRow({ children: [new TableCell({ width: { size: 5000, type: WidthType.PERCENTAGE }, columnSpan: 2, borders: bdrs, margins: cMar, verticalAlign: VerticalAlign.CENTER,
      children: [new Paragraph({ children: [sb("2. \u8BFE\u7A0B\u5B66\u4E60\u60C5\u51B5")] })] })] }),
    new TableRow({ children: [
      new TableCell({ width: { size: C1, type: WidthType.DXA }, borders: bdrs, margins: cMar, verticalAlign: VerticalAlign.CENTER,
        children: [new Paragraph({ children: [sr("\u662F\u5426\u5DF2\u8FBE\u5230\u57F9\u517B\u65B9\u6848\u89C4\u5B9A\u7684\u5B66\u5206\u8981\u6C42")] })] }),
      new TableCell({ width: { size: C2, type: WidthType.DXA }, borders: bdrs, margins: cMar, verticalAlign: VerticalAlign.CENTER,
        children: [new Paragraph({ children: [sr("\u2611\u662F    \u25A1\u5426")] })] })
    ] }),
    new TableRow({ children: [new TableCell({ width: { size: 5000, type: WidthType.PERCENTAGE }, columnSpan: 2, borders: bdrs, margins: cMar, verticalAlign: VerticalAlign.CENTER,
      children: [new Paragraph({ children: [sb("3. \u8BBA\u6587\u7814\u7A76\u8FDB\u5C55")] })] })] }),
    new TableRow({ children: [new TableCell({ width: { size: 5000, type: WidthType.PERCENTAGE }, columnSpan: 2, borders: bdrs, margins: cMar,
      children: content })] })
  ]
});

const body = [
  new Paragraph({ children: [hr("\u4E00\u3001\u5DF2\u5B8C\u6210\u7684\u4E3B\u8981\u5DE5\u4F5C", { size: 28 })] }),
  mainTable
];

const doc = new Document({
  sections: [
    { properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1418, right: 1191, bottom: 1418, left: 1474 } } }, children: cover },
    { properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1418, right: 1191, bottom: 1418, left: 1474 } } },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT] })] })] }) },
      children: body }
  ]
});

Packer.toBuffer(doc).then(buf => {
  const out = path.join(__dirname, "\u4E2D\u671F\u62A5\u544A_\u5371\u9669\u56E0\u7D20\u5206\u6790.docx");
  fs.writeFileSync(out, buf);
  console.log("OK: " + out + " (" + (buf.length / 1024).toFixed(1) + " KB)");
});
