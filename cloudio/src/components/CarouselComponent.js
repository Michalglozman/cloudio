import React from 'react';
import { Carousel } from 'react-responsive-carousel';
import '../style/carouselStyle.css'; // Import the CSS file

const CarouselComponent = ({ downloadedImage, predictedFile, predictedMaksdFile}) => {
  return (
    <Carousel
      showThumbs={false}
      showArrows={true}
      dynamicHeight={true}
      renderArrowPrev={(onClickHandler, hasPrev) => (
        hasPrev && (
          <button
            type="button"
            onClick={onClickHandler}
            className="carousel-button carousel-button-prev"
          >
            &lt;
          </button>
        )
      )}
      renderArrowNext={(onClickHandler, hasNext) => (
        hasNext && (
          <button
            type="button"
            onClick={onClickHandler}
            className="carousel-button carousel-button-next"
          >
            &gt;
          </button>
        )
      )}
      renderIndicator={(onClickHandler, isSelected, index, label) => {
        const indicatorClass = isSelected
          ? 'carousel-indicator carousel-indicator-selected'
          : 'carousel-indicator carousel-indicator-default';

        return (
          <button
            type="button"
            className={indicatorClass}
            onClick={onClickHandler}
            onKeyDown={onClickHandler}
            value={index}
            key={index}
            role="button"
            tabIndex={0}
            title={`${label} ${index + 1}`}
            aria-label={`${label} ${index + 1}`}
          />
        );
      }}
    >
      {downloadedImage && (
        <div>
          <img src={`data:image/png;base64,${downloadedImage}`} alt="Downloaded Image" />
        </div>
      )}

      {predictedFile && (
        <div>
          <img src={`data:image/png;base64,${predictedFile}`} alt="Predicted Image" />
        </div>
      )}

      {predictedMaksdFile && (
        <div>
          <img src={`data:image/png;base64,${predictedMaksdFile}`} alt="Predicted Masked Image" />
        </div>
      )}
    </Carousel>
  );
};

export default CarouselComponent;
