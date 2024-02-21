from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import AllowAny

from .process_word_60k import predTextWord60K
from .process_word_30k import predTextWord30K
from .process_word_piece import predTextWordPiece
from .process_sentence_piece import predTextSentencePiece
from .process_bpe import predTextBPE
from .process_morpheme import predTextMorpheme
from .process_morpheme_bpe import predTextMorphemeBPE

class TransformerLmView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        """
            Sentimental Classification of the input string
        """
        input_text = request.data.get('body')
        num_words = int(request.data.get('num_words'))
        word_60k_prediction = predTextWord60K(input_text, num_words)
        word_30k_prediction = predTextWord30K(input_text, num_words)
        pred_word_piece = predTextWordPiece(input_text, num_words)

        pred_sentence_piece = predTextSentencePiece(input_text, num_words)
        pred_bpe = predTextBPE(input_text, num_words)
        pred_morpheme = predTextMorpheme(input_text, num_words)
        pred_morpheme_bpe = predTextMorphemeBPE(input_text, num_words)

        response_dict = {"Input String": input_text, "Word60kModel": word_60k_prediction, "Word30kModel" : word_30k_prediction, "word_piece_prediction" : pred_word_piece, "sentence_piece_prediction" : pred_sentence_piece, "bpe_prediction" : pred_bpe, "morpheme_prediction" : pred_morpheme, "morpheme_bpe_prediction" : pred_morpheme_bpe}
        
        
        return Response(response_dict)