"""OCR-assisted UI transitions. Gameplay target selection is handled separately."""
import re
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR


@lru_cache(maxsize=1)
def pause_template():
    return np.asarray(Image.open(Path(__file__).parent/'assets/paused-mask.png'))>0


class GameUI:
    def __init__(self):
        self.ocr = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=2)

    def read(self, rgb):
        result, _ = self.ocr(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return [dict(text=row[1], confidence=float(row[2]),
                     center=np.asarray(row[0]).mean(axis=0).tolist()) for row in result or []]

    @staticmethod
    def is_paused(rgb):
        patch=rgb[175:270,200:445].astype(float)
        warm=(patch[:,:,0]>170)&(patch[:,:,1]>65)&(patch[:,:,2]<patch[:,:,0]*.75)
        template=pause_template()
        intersection=np.count_nonzero(warm&template)
        return bool(2*intersection/max(1,np.count_nonzero(warm)+np.count_nonzero(template))>.85)

    @staticmethod
    def panel_hint(rgb):
        hsv = cv2.cvtColor(rgb[378:403, 150:495], cv2.COLOR_RGB2HSV)
        return float(np.mean((hsv[:,:,0] > 35) & (hsv[:,:,0] < 85) & (hsv[:,:,1] > 65) & (hsv[:,:,2] > 75))) > .65

    @staticmethod
    def ready_hint(rgb):
        hsv=cv2.cvtColor(rgb[174:194,210:440],cv2.COLOR_RGB2HSV)
        return float(np.mean((hsv[:,:,0]>35)&(hsv[:,:,0]<85)&(hsv[:,:,1]>50)&(hsv[:,:,2]>75)))>.65

    @staticmethod
    def classify(rows):
        texts = [re.sub(r'[^A-Z0-9]', '', row['text'].upper()) for row in rows]
        joined = ' '.join(texts)
        def find(words):
            for row, text in zip(rows, texts):
                if text in words and row['confidence'] > .55:
                    return row['center']
            return None
        if 'STATS' in texts and ('POINTS' in texts or 'COMBOS' in texts):
            return 'level_won', find({'OK'}) or [320, 390]
        if any('GETREADY' in text for text in texts):
            return 'get_ready',find({'GO','G0'}) or [320,183]
        if any(t.startswith('PAUS') and len(t) <= 7 for t in texts):
            return 'paused', None
        if 'GAMEOVER' in ''.join(texts) or any(t.startswith('GAME') and t.endswith('VER') for t in texts):
            return 'game_over', find({'TRYAGAIN', 'PLAYAGAIN', 'RETRY', 'CONTINUE', 'OK'})
        if any('ENTERNEXTTEMPLE' in t for t in texts):
            return 'temple_confirmation', find({'OK'}) or [234,220]
        if 'ADVENTURE' in texts and 'GAUNTLET' in texts:
            return 'main_menu', find({'ADVENTURE'})
        if 'PLAY' in texts and ('STAGE' in joined or 'TEMPLE' in joined):
            return 'stage_menu', find({'PLAY'})
        if any(t.startswith('INSTR') and t.endswith('TIONS') for t in texts):
            return 'instructions', find({'OK'})
        if any('RIBBIT' in t or 'INTERLOPER' in t or 'YOURTHINKING' in t for t in texts):
            return 'chapter_dialogue', [550,430]
        if 'ZUMA' in joined and ('CONGRATULATIONS' in joined or 'VICTORY' in joined):
            return 'possible_campaign_end', None
        return 'unknown', None

    def hud(self, image):
        rows = self.read(cv2.resize(image[:28, 145:245], (400, 112)))
        text = ''.join(row['text'].upper() for row in rows)
        raw = text
        text = re.sub(r'^[IL1]?V[IL1]\s*', '', text.replace('LEVEL', ''))
        cleaned = text.replace(' ', '').replace('I', '1').replace('L', '1').replace('H', '1')
        match = re.search(r'(\d{1,2})[-:](\d)', cleaned)
        level = [int(match[1]), int(match[2])] if match else None
        if level and not (1 <= level[0] <= 13 and 1 <= level[1] <= 7):
            level = None
        return dict(raw=raw, level=level)

    def gauntlet_hud(self, image):
        rows=self.read(cv2.resize(image[:28,145:248],(412,112)))
        raw=''.join(row['text'].upper() for row in rows)
        text=re.sub(r'[^A-Z0-9+]','',raw)
        match=re.search(r'(RAB(?:BIT)?|EAG(?:LE)?|JAG(?:UAR)?|SUN(?:GOD)?|SG)([0-9IL]+)',text)
        ranks={'RAB':0,'EAG':1,'JAG':2,'SUN':3,'SG':3}
        return dict(raw=raw,rank=ranks[match[1][:3]] if match else None,
                    round=int(match[2].replace('I','1').replace('L','1')) if match else None)
