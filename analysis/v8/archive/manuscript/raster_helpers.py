"""Build one raster PNG per manuscript figure from the original PPT assets.

Figure 1 is copied unchanged. All other lettering, cleaning and panel placement
is completed in PNG pixels before LaTeX inclusion; no vector overlays are used.
No model inference, clustering or scientific-coordinate reconstruction occurs.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import hashlib,json,shutil
M=Path(__file__).resolve().parent;OUT=M/'figures_composed';OUT.mkdir(exist_ok=True)
LAYOUT=json.loads((M/'raster_layout.json').read_text());DPI=600;WIDTH=3300
FONT=M/'assets/DejaVuSans.ttf';BOLD=M/'assets/DejaVuSans-Bold.ttf'
for f in [FONT,BOLD]:
 if not f.exists():raise FileNotFoundError(f)
font=ImageFont.truetype(str(FONT),round(8*DPI/72));bold=ImageFont.truetype(str(BOLD),round(9*DPI/72))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def render(name,width):
 c=LAYOUT[name];src=M/'figures'/f'{name}.png';raw=Image.open(src).convert('RGBA');im=Image.new('RGB',raw.size,'white');im.paste(raw,mask=raw.getchannel('A'));w,h=im.size;draw=ImageDraw.Draw(im)
 for box in c.get('masks',[]):
  a,b,d,e=box['rect'];draw.rectangle((round(a*w),round(b*h),round(d*w),round(e*h)),fill='#'+box.get('color','FFFFFF'))
  for line in box.get('restore_lines',[]):draw.line((round(a*w),round(line['y']*h),round(d*w),round(line['y']*h)),fill='#'+line['color'],width=1)
 x0,y0,x1,y1=c['crop'];left,top,right,bottom=round(x0*w),round(y0*h),round(x1*w),round(y1*h)
 view=im.crop((left,top,right,bottom));scale=width/view.width;pad=round(c.get('pad_top_bp',0)*DPI/72)
 view=view.resize((width,round(view.height*scale)),Image.Resampling.LANCZOS)
 result=Image.new('RGB',(width,view.height+pad),'white');result.paste(view,(0,pad));draw=ImageDraw.Draw(result)
 xy=lambda p:((p[0]*w-left)*scale,(p[1]*h-top)*scale+pad)
 for t in c.get('titles',[]):draw.text(xy(t['xy']),t['text'].replace('--','\u2013'),font=font,fill='black',anchor='mt')
 for letter,pos in zip(c['letters'],c['positions']):draw.text(xy(pos),letter,font=bold,fill='black',anchor='lt')
 return result,{'source_asset':str(src.relative_to(M)),'source_sha256':sha(src),'source_size':[w,h],'layout':c,'render_size':list(result.size)}
