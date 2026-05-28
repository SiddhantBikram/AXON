import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Optional, List, Dict, Any

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(arch: str = "ViT-B/16", design_details: Optional[Dict] = None):
    """
    Load CLIP model to CPU.
    
    Args:
        arch: CLIP architecture name
        design_details: Design details for custom CLIP build
        
    Returns:
        CLIP model
    """
    url = clip._MODELS[arch]
    model_path = clip._download(url)
    
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    
    if design_details is None:
        design_details = {
            "trainer": 'AXON',
            "vision_depth": 0,
            "language_depth": 0,
            "vision_ctx": 0,
            "language_ctx": 4
        }
    
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model


class TextEncoder(nn.Module):
    """CLIP Text Encoder wrapper."""
    
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
    
    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        
        # Take features from the eot embedding
        selected_x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
        selected_x = selected_x.to(self.text_projection.dtype)
        x = selected_x @ self.text_projection
        
        return x


class ResidualFeatureDistillation(nn.Module):
    """
    Residual Feature Distillation module for bridging pose and text features.
    """
    
    def __init__(self, input_dim: int, output_dim: int, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim, bias=False),
            nn.GELU(),
            nn.Linear(input_dim, output_dim, bias=False)
        )
        # Initialize for stability
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.kaiming_normal_(self.mlp[0].weight)
    
    def forward(self, x):
        return x + self.alpha * self.mlp(x)


class VLPromptLearner(nn.Module):
    """
    Vision-Language Prompt Learner for CLIP.
    
    Supports:
    - Zero-shot evaluation
    - Learnable text prompts
    - Fixed prompts
    """
    
    def __init__(
        self,
        classnames: List[str],
        clip_model,
        use_prompt: bool = False,
        ctx_init: str = "a photo of a",
        n_ctx: int = 4,
        prompt_depth: int = 0,
        zero_shot: bool = False
    ):
        super().__init__()
        dtype = clip_model.dtype
        self.use_prompt_stage = use_prompt
        
        if zero_shot:
            # Zero-shot evaluation
            text_aug = f"{{}}"
            tokenized_prompts = torch.cat([
                clip.tokenize(text_aug.format(c), context_length=77) 
                for c in classnames
            ])
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype).cuda()
            self.register_buffer("complete_text_embeddings", embedding)
            self.tokenized_prompts = tokenized_prompts
            
        elif use_prompt:
            # Learnable prompts
            n_cls = len(classnames)
            ctx_dim = clip_model.ln_final.weight.shape[0]
            
            if ctx_init and n_ctx <= 4:
                ctx_init = ctx_init.replace("_", " ")
                prompt = clip.tokenize(ctx_init)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors = embedding[0, 1:1 + n_ctx, :]
                prompt_prefix = ctx_init
            else:
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = " ".join(["X"] * n_ctx)
            
            self.ctx = nn.Parameter(ctx_vectors)
            classnames = [name.replace("_", " ") for name in classnames]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
            
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            
            self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
            self.n_cls = n_cls
            self.tokenized_prompts = tokenized_prompts
            
        else:
            # Fixed prompts
            ctx_init = ctx_init.replace("_", " ")
            prompts = [ctx_init + " " + name + "." for name in classnames]
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
            
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            
            self.register_buffer("complete_text_embeddings", embedding)
            self.tokenized_prompts = tokenized_prompts
    
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        """Construct prompts from context, prefix, and suffix."""
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts
    
    def forward(self):
        if self.use_prompt_stage:
            ctx = self.ctx
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
            prompts = self.construct_prompts(ctx, self.token_prefix, self.token_suffix)
        else:
            prompts = self.complete_text_embeddings
        return prompts


class Classifier(nn.Module):
    """Base classifier with normalized weights."""
    
    def __init__(self, feat_dim: int = 768, num_classes: int = None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, feat_dim, dtype=dtype))
        self.weight.data.uniform_(-1, 1).renorm_(2, 0, 1e-5).mul_(1e5)
    
    @property
    def dtype(self):
        return self.weight.dtype
    
    def forward(self, x):
        raise NotImplementedError
    
    def apply_weight(self, weight):
        """Apply pre-computed weights (e.g., from text features)."""
        self.weight.data = weight.clone()


class LinearClassifier(Classifier):
    """Linear classifier with bias."""
    
    def __init__(self, feat_dim: int = None, num_classes: int = None, dtype=None, **kwargs):
        super().__init__(feat_dim, num_classes, dtype)
        nn.init.kaiming_normal_(self.weight.data)
        self.bias = nn.Parameter(torch.zeros(num_classes, dtype=dtype))
    
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


def logsum_distance(student_z: torch.Tensor, teacher_z: torch.Tensor) -> torch.Tensor:
    """
    Compute LogSum distance loss for feature distillation.
    
    This is a practical implementation of the soft maximum concept
    for aligning student and teacher representations.
    
    Args:
        student_z: Student features
        teacher_z: Teacher features
        
    Returns:
        LogSum distance loss
    """
    diff = student_z - teacher_z
    loss = torch.logsumexp(diff, dim=-1)
    return loss.mean()


class AXON(nn.Module):
    
    def __init__(
        self,
        classnames: List[str],
        clip_model,
        use_prompt: bool = False,
        hyperformer_model: Optional[nn.Module] = None,
        feature_dim: int = 512,
        num_classes: int = 6
    ):
        super().__init__()
        
        # Text encoding components
        self.prompt_learner = VLPromptLearner(
            classnames, clip_model,
            use_prompt=use_prompt
        )
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        
        # Freeze text encoder
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
        
        # Projection and classification
        self.projector = nn.Linear(feature_dim, feature_dim, bias=False)
        self.classifier = LinearClassifier(feature_dim, num_classes, self.text_encoder.dtype)
        
        # Batch normalization for feature alignment
        self.student_bn = nn.BatchNorm1d(feature_dim, affine=False, eps=0.0001)
        self.teacher_bn = nn.BatchNorm1d(feature_dim, affine=False, eps=0.0001)
        
        # Hyperformer for pose encoding
        self.hyperformer_model = hyperformer_model
        
        self.class_names = classnames
        
        # Initialize classifier with text features
        self._init_head_text_feat()
    
    def _init_head_text_feat(self):
        """Initialize classifier head with text features."""
        print("Initializing classifier head with text features...")
        tokenized_prompts = self.tokenized_prompts
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = F.normalize(text_features, dim=-1)
        self.classifier.apply_weight(text_features)
    
    def forward(
        self,
        pose_data: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Forward pass.
        
        Args:
            pose_data: Skeleton/pose data [B, C, T, V, M]
            
        Returns:
            If pose_data provided: (distilled_embeddings, logits, text_features)
            Otherwise: text_features
        """
        tokenized_prompts = self.tokenized_prompts
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts)
        
        if pose_data is not None:
            # Encode pose
            pose_embedding = self.hyperformer_model(pose_data)
            pose_embedding_distilled = self.projector(pose_embedding)
            
            # Classification
            pose_logits = self.classifier(pose_embedding)
            
            # Normalize for distillation
            pose_embedding_distilled = self.student_bn(pose_embedding_distilled)
            text_features_norm = self.teacher_bn(text_features)
            
            return pose_embedding_distilled, pose_logits, text_features_norm
        else:
            return text_features
    
    @staticmethod
    def import_class(name: str):
        """Dynamically import a class by name."""
        components = name.split('.')
        mod = __import__(components[0])
        for comp in components[1:]:
            mod = getattr(mod, comp)
        return mod
    
    @classmethod
    def load_hyperformer(
        cls,
        model_class: str,
        weights_path: Optional[str] = None,
        config: Any = None
    ) -> nn.Module:
        """
        Load Hyperformer model.
        
        Args:
            model_class: Full class path for Hyperformer
            weights_path: Path to pretrained weights
            config: Config object for model initialization
            
        Returns:
            Loaded Hyperformer model
        """
        HypModel = cls.import_class(model_class)
        hyp_model = HypModel(config) if config else HypModel()
        
        if weights_path and os.path.exists(weights_path):
            print(f"Loading Hyperformer weights from {weights_path}")
            
            if '.pkl' in weights_path:
                with open(weights_path, 'rb') as f:
                    weights = pickle.load(f)
            else:
                chkpnt = torch.load(weights_path, map_location='cpu')
                weights = chkpnt
            
            # Clean up keys
            weights = OrderedDict([
                [k.split('module.')[-1], v.cuda()] 
                for k, v in weights.items()
            ])
            
            try:
                hyp_model.load_state_dict(weights, strict=False)
                print("Loaded Hyperformer weights successfully")
            except Exception as e:
                print(f"Partial weight loading: {e}")
                state = hyp_model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print(f"Missing weights: {diff}")
                state.update(weights)
                hyp_model.load_state_dict(state, strict=False)
        
        return hyp_model


def build_model(
    classnames: List[str],
    arch: str = "ViT-B/16",
    use_prompt: bool = False,
    hyperformer_class: Optional[str] = None,
    hyperformer_weights: Optional[str] = None,
    config: Any = None,
    freeze_text: bool = True
) -> AXON:

    print(f"Loading CLIP (backbone: {arch})")
    clip_model = load_clip_to_cpu(arch)
    
    # Load Hyperformer if specified
    hyperformer_model = None
    if hyperformer_class:
        hyperformer_model = AXON.load_hyperformer(
            hyperformer_class, hyperformer_weights, config
        )
    
    print("Building AXON model")
    model = AXON(
        classnames=classnames,
        clip_model=clip_model,
        use_prompt=use_prompt,
        hyperformer_model=hyperformer_model
    )
    
    # Configure gradients
    if freeze_text:
        for name, param in model.named_parameters():
            if "text_encoder" in name:
                param.requires_grad = False
    
    # Print trainable parameters
    enabled = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(name)
    print(f"Trainable parameters: {len(enabled)}")
    
    model.float()
    return model
