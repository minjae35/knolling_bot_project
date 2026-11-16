const ids=[];
await Promise.all(['Regular','SemiBold','Italic'].map(style=>figma.loadFontAsync({family:'Source Sans 3',style})));
const C={ink:'#20272D',muted:'#687580',blue:'#245E8A',pale:'#E9F1F7',rule:'#896441',line:'#C8D0D5',white:'#FFFFFF'};
const paint=c=>[{type:'SOLID',color:figma.util.rgb(c)}];
const root=figma.createAutoLayout('VERTICAL');root.name='Figure 2 — Hybrid inference and learned structure';root.resize(396,260);root.layoutSizingHorizontal='FIXED';root.layoutSizingVertical='FIXED';root.itemSpacing=0;root.fills=paint(C.white);root.x=200;root.y=100;ids.push(root.id);
const canvas=figma.createFrame();canvas.name='Editable scientific diagram';canvas.resize(396,260);canvas.fills=[];root.appendChild(canvas);ids.push(canvas.id);
function txt(s,x,y,size=8,style='Regular',col=C.ink,w){let n=figma.createText();n.name=s;canvas.appendChild(n);n.fontName={family:'Source Sans 3',style};n.fontSize=size;n.lineHeight={unit:'PIXELS',value:size*1.18};n.characters=s;n.fills=paint(col);if(w){n.textAutoResize='HEIGHT';n.resize(w,n.height);}n.x=x;n.y=y;ids.push(n.id);return n;}
function rect(name,x,y,w,h,fill,stroke=C.line,sw=.6){let n=figma.createRectangle();n.name=name;canvas.appendChild(n);n.resize(w,h);n.x=x;n.y=y;n.fills=fill?paint(fill):[];n.strokes=stroke?paint(stroke):[];n.strokeWeight=sw;ids.push(n.id);return n;}
function path(name,d,c=C.ink,lw=.7,dash){let n=figma.createVector();n.name=name;canvas.appendChild(n);n.vectorPaths=[{windingRule:'NONZERO',data:d}];n.fills=[];n.strokes=paint(c);n.strokeWeight=lw;if(dash)n.dashPattern=dash;ids.push(n.id);return n;}
function arrow(name,d,x,y,dir='right',col=C.ink){path(name,d,col,.72);const p={right:`M ${x-3} ${y-1.8} L ${x} ${y} L ${x-3} ${y+1.8}`,down:`M ${x-1.8} ${y-3} L ${x} ${y} L ${x+1.8} ${y-3}`,up:`M ${x-1.8} ${y+3} L ${x} ${y} L ${x+1.8} ${y+3}`}[dir];path(name+' arrowhead',p,col,.85);}
function token(x,y,w=4,h=7,col=C.blue){return rect('Schematic feature token',x,y,w,h,C.pale,col,.45);}
txt('a',0,0,9,'SemiBold');txt('Hybrid inference',13,0,9,'SemiBold');
const rgb=rect('Original model input RGB',0,22,45,30,'#EEF0F2',null);txt('RGB image',0,55,7.5,'Regular',C.muted);
arrow('RGB to detector','M 49 37 L 62 37',62,37);
rect('Detector layer 3',68,21,73,28,C.white,C.blue,.6);rect('Detector layer 2',66,23,73,28,C.white,C.blue,.6);rect('Detector layer 1',64,25,73,28,C.white,C.blue,.6);
txt('RT-DETR-R18',70,28,8,'SemiBold',C.blue);txt('+ ROI refinements',70,40,7.3,'Regular',C.muted);
arrow('Detector to object features','M 143 37 L 154 37',154,37);
for(let r=0;r<3;r++)for(let c=0;c<12;c++)token(159+c*3.7,23+r*8,2.7,6);
txt('O : N × 12',164,51,8,'Italic',C.blue);
arrow('Objects to layout','M 205 37 L 220 37',220,37);
txt('Layout proposal',227,22,8.2,'SemiBold');
for(let r=0;r<2;r++){for(let c=0;c<3;c++)token(229+c*15,38+r*10,8,6);arrow('Schematic pointer order',`M 239 ${41+r*10} L 243 ${41+r*10}`,243,41+r*10);arrow('Schematic pointer order',`M 254 ${41+r*10} L 258 ${41+r*10}`,258,41+r*10);}
txt('rows · order · initial poses',217,60,7.3,'Regular',C.muted);
arrow('Proposal to geometry','M 281 41 L 311 41',311,41);
rect('Geometric processing',316,22,79,40,null,C.rule,.75);txt('Geometry',327,25,8.5,'SemiBold',C.rule);txt('rank · constrain',327,38,7.4,'Regular',C.rule);txt('repair · merge',327,48,7.4,'Regular',C.rule);
arrow('Geometry to target poses','M 355 64 L 355 79',355,79,'down',C.rule);txt('Target poses P',328,83,8.4,'SemiBold');txt('(x, y, θ) per object',327,95,7.3,'Regular',C.muted);
txt('English text',0,81,8,'Regular');arrow('Text to goal encoder','M 62 86 L 78 86',78,86);
path('Instruction encoding baseline','M 83 102 L 206 102',C.blue,.75);txt('MiniLM + instruction encoder',83,77,8,'SemiBold',C.blue);txt('global goal g (13) · selection m',83,89,7.4,'Regular',C.muted);
arrow('Object-conditioned referents','M 182 64 L 182 72 L 174 72 L 174 76',174,76,'down',C.blue);
arrow('Instruction conditions proposal','M 211 86 L 250 86 L 250 74',250,74,'up',C.blue);
path('Held bypass','M 206 70 L 301 70 L 301 110 L 354 110 L 354 103',C.muted,.65,[2,2]);
txt('held / no-op: detected xy + RGB/PCA angle',1,109,7.4,'Regular',C.muted);
path('Panel separator','M 0 124 L 396 124',C.line,.65);
txt('b',0,133,9,'SemiBold');txt('Learned layout: context selects structure; original features predict poses',13,133,8.6,'SemiBold');
txt('O',0,175,10,'Italic',C.blue);arrow('Object input','M 11 181 L 25 181',25,181,'right',C.blue);
path('Relation layer stack','M 29 163 L 87 163 L 87 194 L 29 194 Z M 33 159 L 91 159 L 91 190',C.blue,.65);
txt('Relation MLP',34,168,8,'SemiBold',C.blue);txt('12 → 128 → 128',33,181,7.3,'Regular',C.muted);
arrow('Relation hidden','M 94 181 L 105 181',105,181,'right',C.blue);txt('h',110,174,11,'Italic',C.blue);
arrow('Hidden to context','M 119 178 L 130 178',130,178,'right',C.blue);
for(let i=0;i<4;i++){rect('Attention head '+(i+1),136+i*14,167,10,23,i%2?C.white:C.pale,C.blue,.55);}
txt('4-head set attention',133,154,7.8,'SemiBold',C.blue);txt('+ goal g + object sizes',131,196,7.2,'Regular',C.muted);
arrow('Context to structure heads','M 192 178 L 208 178',208,178,'right',C.blue);
path('Context head split','M 208 161 L 208 190 M 208 161 L 222 161 M 208 190 L 222 190',C.blue,.65);
txt('Row assignment',225,151,7.9,'SemiBold',C.blue);txt('8 logits / object',225,162,7.2,'Regular',C.muted);
txt('Pointer order',225,181,7.9,'SemiBold',C.blue);txt('autoregressive',225,192,7.2,'Regular',C.muted);
arrow('Discrete structure to assembly','M 288 160 L 309 160 L 309 208',309,208,'down',C.ink);path('Pointer to discrete join','M 282 191 L 309 191',C.ink,.65);txt('r, π',318,179,9,'Italic');
path('Original h bypass','M 114 189 L 114 225 L 285 225',C.blue,.9);arrow('Original features to pose heads','M 285 225 L 295 225',295,225,'right',C.blue);txt('original h · goal · sizes',139,213,7.5,'Regular',C.blue);
path('Pose head underline','M 299 239 L 394 239',C.blue,.85);txt('Pose heads + assembly',300,216,8.0,'SemiBold',C.blue);txt('row origins · steps · angle',300,228,7.1,'Regular',C.muted);
path('Legend learned mark','M 0 254 L 10 254',C.blue,1);txt('learned parameters (staged training)',14,248,7.5,'Regular',C.muted);
path('Legend rule mark','M 220 254 L 230 254',C.rule,1);txt('geometric / explicit rules',234,248,7.5,'Regular',C.muted);
root.exportSettings=[{format:'PDF'},{format:'SVG',svgOutlineText:false,svgIdAttribute:true}];
return {rootId:root.id,rgbNodeId:rgb.id,createdNodeIds:ids,counts:{text:canvas.findAllWithCriteria({types:['TEXT']}).length,vector:canvas.findAllWithCriteria({types:['VECTOR']}).length},width:root.width,height:root.height,font:'Source Sans 3'};
