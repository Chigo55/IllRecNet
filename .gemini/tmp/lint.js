const fs = require('fs');
const path = require('path');

function walk(dir) {
  let results = [];
  const list = fs.readdirSync(dir);
  list.forEach(file => {
    file = path.join(dir, file);
    const stat = fs.statSync(file);
    if (stat && stat.isDirectory()) {
      results = results.concat(walk(file));
    } else if (file.endsWith('.md')) {
      results.push(file.replace(/\\/g, '/'));
    }
  });
  return results;
}

const wikiFiles = walk('.wiki').filter(f => !f.includes('index.md') && !f.includes('log.md') && !f.includes('rules/docs.md') && !f.includes('config.json'));
const indexContent = fs.readFileSync('.wiki/index.md', 'utf8');

const indexLinks = [...indexContent.matchAll(/\[\[([^\]]+)\]\]/g)].map(m => m[1]);

let errors = [];
let warns = [];

// 1. Orphan pages
wikiFiles.forEach(file => {
  const linkPath = file.replace('.md', '');
  if (!indexLinks.includes(linkPath)) {
    errors.push(`[고아 페이지] ${file} -> index.md 미등록`);
  }
});

// 2. Broken links & 4. Unlinked pages
wikiFiles.forEach(file => {
  const content = fs.readFileSync(file, 'utf8');
  const links = [...content.matchAll(/\[\[([^\]]+)\]\]/g)].map(m => m[1]);
  links.forEach(link => {
    const targetFile = link + '.md';
    if (!fs.existsSync(targetFile)) {
      errors.push(`[깨진 링크] ${file} -> ${link} (파일 없음)`);
    }
  });
  
  if (!content.includes('## 연결')) {
    warns.push(`[연결 없음] ${file} -> ## 연결 섹션 없음`);
  } else {
    const lines = content.split('\n');
    const idx = lines.findIndex(l => l.startsWith('## 연결'));
    const nextLines = lines.slice(idx + 1).filter(l => l.trim().startsWith('- [['));
    if (nextLines.length === 0) {
      warns.push(`[연결 없음] ${file} -> 연결된 링크 없음`);
    }
  }

  // 3. Outdated pages
  const match = content.match(/\*\*갱신\*\*: ([\d-]+)/);
  if (match) {
    const date = new Date(match[1]);
    const now = new Date();
    const diff = (now - date) / (1000 * 60 * 60 * 24);
    if (diff > 30) {
      warns.push(`[오래된 페이지] ${file} (갱신: ${match[1]}, ${Math.floor(diff)}일 경과)`);
    }
  }
});

const date = new Date().toISOString().split('T')[0];
console.log(`Wiki Lint Report — ${date}`);
console.log(`──────────────────────────────`);
if (errors.length > 0) {
  console.log(`ERROR (${errors.length})`);
  errors.forEach(e => console.log(`  ${e}`));
  console.log('');
}
if (warns.length > 0) {
  console.log(`WARN (${warns.length})`);
  warns.forEach(w => console.log(`  ${w}`));
  console.log('');
}
console.log(`INFO`);
console.log(`  전체 페이지: ${wikiFiles.length}개 | 점검 완료: ${wikiFiles.length}개`);
console.log(`──────────────────────────────`);
console.log(`총 이슈: ERROR ${errors.length}, WARN ${warns.length}`);
