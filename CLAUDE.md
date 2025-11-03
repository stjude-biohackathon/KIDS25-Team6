# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MolSnap is a full-stack AI application that converts chemical structure images (from PDFs or standalone images) into SMILES (Simplified Molecular Input Line Entry System) strings. The project uses computer vision and sequence generation to enable chemists to digitize chemical structures for computational analysis.

## Architecture

The application follows a three-tier microservices architecture:

- **Frontend**: React 19 + TypeScript + Vite (Port 5173)
- **Document Processing API**: FastAPI (Port 8001) - Extracts chemical structures from PDFs
- **ML Prediction API**: FastAPI (Port 8000) - Converts images to SMILES using MolNexTR model
- **ML Model**: MolNexTR (Vision Transformer + LSTM decoder) for image-to-SMILES conversion

## Development Commands

### Frontend Development (molsnap/)
```bash
cd molsnap
npm install                # Install dependencies
npm run dev               # Start dev server at http://localhost:5173
npm run build            # Build for production (TypeScript compile + Vite build)
npm run lint             # Run ESLint
npm run preview          # Preview production build
```

### Backend Development
```bash
# Environment setup
conda env create -n molsnap python=3.10
conda activate molsnap
pip install -r requirements.txt

# Download ML model checkpoint (required)
# Place molnextr_best.pth in checkpoints/ folder
wget https://zenodo.org/records/13304899/files/molnextr_best.pth

# Start document processing API (Port 8001)
cd decimer-api
uvicorn main:app --reload --port 8001

# Start ML prediction API (Port 8000)
cd molsnap-api
uvicorn main:app --reload

# Alternative: Run APIs directly
python molsnap-api/main.py
python decimer-api/main.py
```

### ML Model Training
```bash
# Fine-tune model (example with test data)
torchrun --nproc_per_node=1 ML_model/train.py \
  --data_path Training-Data/test_mini \
  --train_file test_mini_train.csv \
  --valid_file test_mini_valid.csv \
  --load_path checkpoints/molnextr_best.pth \
  --save_path output_path \
  --epochs 40 \
  --batch_size 32 \
  --do_train \
  --do_valid \
  --fp16
```

### Model Inference (Python)
```python
from ML_model import prediction
results = prediction.predict_from_image_files(['image1.png', 'image2.png'], 'checkpoints/molnextr_best.pth')
```

## Key File Structure

```
├── molsnap/                          # React frontend
│   ├── src/
│   │   ├── pages/                    # Main application pages
│   │   ├── components/               # Reusable UI components
│   │   ├── context/                  # React Context state management
│   │   ├── utils/                    # API utilities
│   │   └── constants.tsx             # API endpoint configuration
│   └── vite.config.ts               # Vite build config with path aliases
├── molsnap-api/                     # ML prediction FastAPI service
│   └── main.py                      # Endpoints: /prediction, /predictions, /prediction-only
├── decimer-api/                     # PDF processing FastAPI service
│   ├── main.py                      # Endpoint: /upload-and-get-chemical-images
│   └── segment.py                   # Chemical structure extraction logic
├── ML_model/                        # MolNexTR model implementation
│   ├── MolNexTR/                    # Core model architecture
│   │   ├── models/                  # PyTorch model definitions
│   │   ├── vocab/                   # SMILES tokenization vocabularies
│   │   └── components.py            # Encoder/Decoder implementations
│   ├── prediction.py                # Inference interface
│   └── train.py                     # Distributed training script
└── checkpoints/                     # Model weights (download required)
```

## State Management (Frontend)

The frontend uses React Context pattern for global state:

- **UploadContext**: File uploads, model selection, extracted images
- **ResultsContext**: Prediction results and display data
- **LoadingContext**: Async operation status tracking

State updates use useReducer pattern with typed actions.

## API Endpoints

### molsnap-api (Port 8000)
- `POST /prediction` - Upload single image → SMILES
- `POST /predictions` - Upload multiple images → SMILES
- `POST /prediction-only` - Predict from pre-extracted image paths
- `GET /` - Health check

### decimer-api (Port 8001)
- `POST /upload-and-get-chemical-images` - Upload PDF → extract chemical structure images
- `GET /` - Health check

## Application Flow

1. **Landing Page** → User clicks "Upload Chemical Structure"
2. **Upload Page** → User uploads PDF/image, selects page range (if PDF)
3. **PDF Processing** → decimer-api extracts chemical structures from PDF pages
4. **Image Selection** → User selects which extracted structures to process
5. **Model Selection** → User chooses ML model checkpoint
6. **Prediction** → molsnap-api processes images through MolNexTR model
7. **Results Page** → Display SMILES strings with confidence scores, export options

## Technology Stack

### Frontend
- React 19.1.1 + TypeScript + Vite 7.1.7
- Material-UI 7.3.2 + Tailwind CSS 4.1.13
- React Router 7.5.3 for routing
- react-pdf 10.1.0 for PDF viewing
- Axios 1.12.2 for API calls

### Backend
- FastAPI 0.115.0 + Uvicorn (async APIs)
- PyTorch + torchvision (ML framework)
- RDKit 2023.9.1 (chemistry library)
- OpenCV + PIL (image processing)
- MinerU (PDF content extraction)

### ML Model
- MolNexTR: Swin Transformer encoder + LSTM decoder
- Input: 384x384 PNG chemical structure images
- Output: SMILES strings + confidence scores
- Training: PyTorch Lightning with distributed training support

## Configuration Files

- `molsnap/src/constants.tsx` - API endpoint URLs
- `molsnap/vite.config.ts` - Build config with path aliases (@components, @pages, @context)
- `molsnap/tsconfig.json` - TypeScript compiler settings
- `ML_model/MolNexTR/vocab/` - SMILES tokenization vocabularies

## Development Patterns

### Frontend
- Use path aliases (@components, @pages, @context) for imports
- Follow React Context + useReducer for state management
- TypeScript strict mode enabled
- Material-UI theming in `src/theme.tsx`

### Backend
- Async FastAPI endpoints with CORS middleware
- File uploads use timestamp-based naming for uniqueness
- Separation of concerns: decimer-api handles PDFs, molsnap-api handles ML predictions

### ML
- Distributed training with PyTorch DDP
- Mixed precision (fp16) training for efficiency
- Data augmentation using Albumentations
- Beam search decoding for SMILES generation

## Important Dependencies

### Python (requirements.txt)
- `torch`, `torchvision` - Deep learning framework
- `transformers` - Hugging Face models
- `rdkit` - Chemistry library for SMILES validation
- `fastapi`, `uvicorn` - Web API framework
- `opencv-python`, `Pillow` - Image processing
- `pytorch-lightning` - Training framework

### Frontend (package.json)
- `react`, `react-dom` - UI framework
- `@mui/material` - Component library
- `tailwindcss` - CSS utility framework
- `react-pdf` - PDF viewing capabilities
- `axios` - HTTP client

## Training Data

The model was trained on ~619K SMILES structures:
- ~270K from LOTUS natural products database
- ~340K from Enamine REAL library (macrocycles, MW > 600 Da)
- Images are 384x384 PNG format
- Full training data located at St. Jude (12GB, see Training-Data/data/training.md)

## Validation

The project includes validation datasets:
- `Validation-Data/real/CLEF/` - CLEF benchmark (450 images)
- `Validation-Data/synthetic/` - Synthetic test structures
- Results stored in `Results/` with performance metrics

## Development Tips

1. **Model checkpoint required**: Download molnextr_best.pth before running ML predictions
2. **CORS setup**: APIs include CORS middleware for frontend-backend communication
3. **Error handling**: Both APIs return structured error responses
4. **File management**: Uploaded files use timestamp naming to prevent conflicts
5. **Context state**: Use React DevTools to debug context state changes
6. **ML debugging**: Model includes confidence scores for prediction quality assessment